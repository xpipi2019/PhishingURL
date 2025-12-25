"""
特征提取模块
"""

import json
import re
import socket
import ssl
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

try:
    import whois
except ImportError:
    whois = None  # type: ignore

from ..utils import logger

# 配置文件路径
_CONFIG_FILE = Path(__file__).parent / "domain_config.json"


def _load_domain_config() -> Dict[str, List[str]]:
    """从JSON文件加载域名配置"""
    try:
        if _CONFIG_FILE.exists():
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                return {
                    "shortening_services": config.get("shortening_services", []),
                    "domain_prefixes": config.get("domain_prefixes", []),
                }
        else:
            logger.warning(f"配置文件不存在: {_CONFIG_FILE}，使用默认配置")
            return {
                "shortening_services": [
                    "bit.ly",
                    "goo.gl",
                    "tinyurl.com",
                    "t.co",
                    "ow.ly",
                    "is.gd",
                    "buff.ly",
                    "short.link",
                    "rebrand.ly",
                    "cutt.ly",
                    "shorturl.at",
                ],
                "domain_prefixes": [
                    "www.",
                    "www1.",
                    "www2.",
                    "www3.",
                    "mail.",
                    "ftp.",
                    "blog.",
                    "shop.",
                ],
            }
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}，使用默认配置")
        return {
            "shortening_services": [
                "bit.ly",
                "goo.gl",
                "tinyurl.com",
                "t.co",
                "ow.ly",
                "is.gd",
                "buff.ly",
                "short.link",
                "rebrand.ly",
                "cutt.ly",
                "shorturl.at",
            ],
            "domain_prefixes": [
                "www.",
                "www1.",
                "www2.",
                "www3.",
                "mail.",
                "ftp.",
                "blog.",
                "shop.",
            ],
        }


# 加载配置
_DOMAIN_CONFIG = _load_domain_config()
SHORTENING_SERVICES = _DOMAIN_CONFIG["shortening_services"]


class FeatureExtractor:
    """
    特征提取器

    从URL中提取30个特征用于钓鱼网站检测。
    包括URL结构特征、域名特征、SSL特征、HTML特征等。

    Attributes:
        session: requests会话对象，用于HTTP请求
        timeout: HTTP请求超时时间（秒）
        _whois_cache: WHOIS查询结果缓存字典
        domain_prefixes: 域名前缀列表（如www, mail等）
    """

    def __init__(self) -> None:
        """
        初始化特征提取器

        创建HTTP会话，设置User-Agent和超时时间。
        加载域名配置（缩短服务列表、域名前缀列表）。
        """
        from ..config import settings
        from ..constants import DEFAULT_HTTP_TIMEOUT, DEFAULT_USER_AGENT

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        self.timeout: int = settings.FEATURE_EXTRACTION_TIMEOUT or DEFAULT_HTTP_TIMEOUT
        self._whois_cache: Dict[str, Optional[Any]] = {}  # WHOIS查询缓存
        # 从配置文件加载域名前缀列表
        self.domain_prefixes: List[str] = _DOMAIN_CONFIG["domain_prefixes"]

    @staticmethod
    def _to_str(value: Any) -> str:
        """将BeautifulSoup属性值安全转换为字符串"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def get_default_features(features_count: int = 30) -> np.ndarray:
        """
        获取默认特征向量（全部为0）

        当无法从URL提取特征时，返回全零向量作为默认值。

        Args:
            features_count: 特征数量，默认30

        Returns:
            np.ndarray: 全零特征向量，形状为 (features_count,)

        Examples:
            >>> features = FeatureExtractor.get_default_features(30)
            >>> print(features)  # [0. 0. 0. ... 0.]
        """
        from ..constants import DEFAULT_FEATURES_COUNT

        if features_count <= 0:
            features_count = DEFAULT_FEATURES_COUNT
        return np.zeros(features_count, dtype=np.float64)

    def extract_url_features(self, url: str) -> np.ndarray:
        """
        从URL提取30个特征向量

        提取的特征包括：
        - URL结构特征（长度、特殊字符等）
        - 域名特征（IP地址、子域名、注册时长等）
        - SSL证书特征
        - HTML内容特征
        - 外部链接特征等

        Args:
            url: 要提取特征的URL字符串

        Returns:
            np.ndarray: 30维特征向量，每个特征值为 -1, 0, 或 1
                - 1: 正常/安全
                - 0: 中性/不确定
                - -1: 可疑/不安全

        Raises:
            ValueError: 当URL格式无效时
            Exception: 当特征提取过程中发生错误时（会返回默认特征向量）

        Examples:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_url_features("https://example.com")
            >>> print(f"提取了 {len(features)} 个特征")
        """
        features = []
        failed_features = []

        # 确保URL有协议
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            path = parsed.path
            port = parsed.port

            # 1. having_IP_Address
            features.append(self._extract_having_ip_address(domain, failed_features))

            # 2. URL_Length
            features.append(self._extract_url_length(url, failed_features))

            # 3. Shortining_Service
            features.append(self._extract_shortening_service(domain, failed_features))

            # 4. having_At_Symbol
            features.append(self._extract_having_at_symbol(url, failed_features))

            # 5. double_slash_redirecting
            features.append(
                self._extract_double_slash_redirecting(path, failed_features)
            )

            # 6. Prefix_Suffix
            features.append(self._extract_prefix_suffix(domain, failed_features))

            # 7. having_Sub_Domain
            features.append(self._extract_having_sub_domain(domain, failed_features))

            # 8. SSLfinal_State
            features.append(self._extract_ssl_final_state(url, domain, failed_features))

            # 9. Domain_registeration_length
            features.append(
                self._extract_domain_registration_length(domain, failed_features)
            )

            # 10. Favicon
            features.append(self._extract_favicon(url, domain, failed_features))

            # 11. port
            features.append(self._extract_port(port, parsed.scheme, failed_features))

            # 12. HTTPS_token
            features.append(self._extract_https_token(domain, failed_features))

            # 13-15. 需要获取HTML内容
            html_content = self._fetch_html(url)
            if html_content:
                # 13. Request_URL
                features.append(
                    self._extract_request_url(html_content, domain, failed_features)
                )
                # 14. URL_of_Anchor
                features.append(
                    self._extract_url_of_anchor(html_content, domain, failed_features)
                )
                # 15. Links_in_tags
                features.append(
                    self._extract_links_in_tags(html_content, domain, failed_features)
                )
                # 16. SFH
                features.append(
                    self._extract_sfh(html_content, domain, failed_features)
                )
                # 17. Submitting_to_email
                features.append(
                    self._extract_submitting_to_email(html_content, failed_features)
                )
                # 20. on_mouseover
                features.append(
                    self._extract_on_mouseover(html_content, failed_features)
                )
                # 21. RightClick
                features.append(
                    self._extract_right_click(html_content, failed_features)
                )
                # 22. popUpWidnow
                features.append(
                    self._extract_popup_window(html_content, failed_features)
                )
                # 23. Iframe
                features.append(self._extract_iframe(html_content, failed_features))
            else:
                # HTML获取失败，填充默认值
                for _ in range(9):
                    features.append(0)
                    failed_features.append("HTML获取失败")

            # 18. Abnormal_URL
            features.append(self._extract_abnormal_url(url, domain, failed_features))

            # 19. Redirect
            features.append(self._extract_redirect(url, failed_features))

            # 24. age_of_domain
            features.append(self._extract_age_of_domain(domain, failed_features))

            # 25. DNSRecord
            features.append(self._extract_dns_record(domain, failed_features))

            # 26. web_traffic
            features.append(self._extract_web_traffic(domain, failed_features))

            # 27. Page_Rank
            features.append(self._extract_page_rank(domain, failed_features))

            # 28. Google_Index
            features.append(self._extract_google_index(domain, failed_features))

            # 29. Links_pointing_to_page
            features.append(
                self._extract_links_pointing_to_page(domain, failed_features)
            )

            # 30. Statistical_report
            features.append(
                self._extract_statistical_report(url, domain, failed_features)
            )

        except Exception as e:
            logger.error(f"提取URL特征时发生错误: {url}, 错误: {str(e)}")
            # 如果发生异常，确保返回30个特征
            while len(features) < 30:
                features.append(0)
                failed_features.append(f"提取过程异常: {e}")

        # 记录无法获取的特征
        if failed_features:
            logger.warning(
                f"URL {url} 无法获取的特征: {', '.join(set(failed_features))}"
            )

        # 确保返回30个特征
        while len(features) < 30:
            features.append(0)
            logger.warning("特征数量不足，补充默认值0")

        return np.array(features[:30], dtype=np.float64)

    def _extract_having_ip_address(self, domain: str, failed_features: list) -> float:
        """提取特征1: having_IP_Address"""
        try:
            # 检查是否为IP地址
            ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
            if re.match(ipv4_pattern, domain):
                return -1.0
            # 尝试解析IPv6
            try:
                socket.inet_pton(socket.AF_INET6, domain)
                return -1.0
            except (socket.error, OSError, ValueError):
                pass
            return 1.0
        except Exception as e:
            failed_features.append("having_IP_Address")
            logger.debug(f"提取having_IP_Address失败: {str(e)}")
            return 0.0

    def _extract_url_length(self, url: str, failed_features: list) -> float:
        """提取特征2: URL_Length"""
        try:
            length = len(url)
            if length < 54:
                return 1.0
            elif length <= 75:
                return 0.0
            else:
                return -1.0
        except Exception as e:
            failed_features.append("URL_Length")
            logger.debug(f"提取URL_Length失败: {str(e)}")
            return 0.0

    def _extract_shortening_service(self, domain: str, failed_features: list) -> float:
        """提取特征3: Shortining_Service"""
        try:
            domain_lower = domain.lower()
            for service in SHORTENING_SERVICES:
                if service in domain_lower:
                    return -1.0
            return 1.0
        except Exception as e:
            failed_features.append("Shortining_Service")
            logger.debug(f"提取Shortining_Service失败: {str(e)}")
            return 0.0

    def _extract_having_at_symbol(self, url: str, failed_features: list) -> float:
        """提取特征4: having_At_Symbol"""
        try:
            return -1.0 if "@" in url else 1.0
        except Exception as e:
            failed_features.append("having_At_Symbol")
            logger.debug(f"提取having_At_Symbol失败: {str(e)}")
            return 0.0

    def _extract_double_slash_redirecting(
        self, path: str, failed_features: list
    ) -> float:
        """提取特征5: double_slash_redirecting"""
        try:
            # 检查path中是否有双斜杠（排除协议部分）
            if "//" in path:
                return -1.0
            return 1.0
        except Exception as e:
            failed_features.append("double_slash_redirecting")
            logger.debug(f"提取double_slash_redirecting失败: {str(e)}")
            return 0.0

    def _extract_prefix_suffix(self, domain: str, failed_features: list) -> float:
        """提取特征6: Prefix_Suffix"""
        try:
            return -1.0 if "-" in domain else 1.0
        except Exception as e:
            failed_features.append("Prefix_Suffix")
            logger.debug(f"提取Prefix_Suffix失败: {str(e)}")
            return 0.0

    def _extract_having_sub_domain(self, domain: str, failed_features: list) -> float:
        """提取特征7: having_Sub_Domain"""
        try:
            dot_count = domain.count(".")
            if dot_count == 1:
                return 1.0
            elif dot_count == 2:
                return 0.0
            else:
                return -1.0
        except Exception as e:
            failed_features.append("having_Sub_Domain")
            logger.debug(f"提取having_Sub_Domain失败: {str(e)}")
            return 0.0

    def _extract_ssl_final_state(
        self, url: str, domain: str, failed_features: list
    ) -> float:
        """提取特征8: SSLfinal_State"""
        try:
            if url.startswith("https://"):
                # 尝试验证SSL证书
                try:
                    context = ssl.create_default_context()
                    with socket.create_connection((domain, 443), timeout=5) as sock:
                        with context.wrap_socket(sock, server_hostname=domain):
                            return 1.0  # 有效HTTPS证书
                except (ssl.SSLError, socket.error, OSError, Exception):
                    return 0.0  # 证书有问题
            else:
                return -1.0  # 无证书
        except Exception as e:
            failed_features.append("SSLfinal_State")
            logger.debug(f"提取SSLfinal_State失败: {str(e)}")
            return 0.0

    def _extract_domain_registration_length(
        self, domain: str, failed_features: list
    ) -> float:
        """提取特征9: Domain_registeration_length"""
        try:
            # 提取纯域名（去掉www等前缀）
            clean_domain = self._extract_clean_domain(domain)
            if not clean_domain:
                failed_features.append("Domain_registeration_length")
                return 0.0

            whois_info = self._get_whois_info(clean_domain)
            if not whois_info:
                failed_features.append("Domain_registeration_length")
                return 0.0

            # 获取注册日期和过期日期
            creation_date = whois_info.get("creation_date")
            expiration_date = whois_info.get("expiration_date")

            if creation_date and expiration_date:
                # 处理可能是列表的情况
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                if isinstance(expiration_date, list):
                    expiration_date = expiration_date[0]

                # 计算注册时长（年）
                if isinstance(creation_date, datetime) and isinstance(
                    expiration_date, datetime
                ):
                    registration_length = (expiration_date - creation_date).days / 365.0
                    # 注册时长 > 1年返回1，否则返回-1
                    return 1.0 if registration_length > 1.0 else -1.0

            failed_features.append("Domain_registeration_length")
            return 0.0
        except Exception as e:
            failed_features.append("Domain_registeration_length")
            logger.debug(f"提取Domain_registeration_length失败: {str(e)}")
            return 0.0

    def _extract_favicon(self, url: str, domain: str, failed_features: list) -> float:
        """提取特征10: Favicon"""
        try:
            html_content = self._fetch_html(url)
            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                # 查找所有link标签
                for link in soup.find_all("link"):
                    rel_attr = link.get("rel")
                    if rel_attr:
                        # rel可能是字符串或列表
                        rel_str = (
                            " ".join(rel_attr)
                            if isinstance(rel_attr, list)
                            else str(rel_attr)
                        )
                        if "icon" in rel_str.lower():
                            href = link.get("href")
                            if href:
                                href_str = (
                                    href[0] if isinstance(href, list) else str(href)
                                )
                                favicon_url = urljoin(url, href_str)
                                favicon_domain = urlparse(favicon_url).netloc
                                if domain in favicon_domain or favicon_domain in domain:
                                    return 1.0
                                else:
                                    return -1.0
            failed_features.append("Favicon")
            return 0.0
        except Exception as e:
            failed_features.append("Favicon")
            logger.debug(f"提取Favicon失败: {str(e)}")
            return 0.0

    def _extract_port(
        self, port: Optional[int], scheme: str, failed_features: list
    ) -> float:
        """提取特征11: port"""
        try:
            if port is None:
                # 默认端口
                return 1.0
            elif port in [80, 443]:
                return 1.0
            else:
                return -1.0
        except Exception as e:
            failed_features.append("port")
            logger.debug(f"提取port失败: {str(e)}")
            return 0.0

    def _extract_https_token(self, domain: str, failed_features: list) -> float:
        """提取特征12: HTTPS_token"""
        try:
            return -1.0 if "https" in domain.lower() else 1.0
        except Exception as e:
            failed_features.append("HTTPS_token")
            logger.debug(f"提取HTTPS_token失败: {str(e)}")
            return 0.0

    def _extract_request_url(
        self, html_content: str, domain: str, failed_features: list
    ) -> float:
        """提取特征13: Request_URL"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            external_count = 0
            total_count = 0

            # 检查图片、视频、音频等资源
            for tag in soup.find_all(["img", "video", "audio", "source"]):
                src = tag.get("src") or tag.get("href")
                if src:
                    total_count += 1
                    src_str = self._to_str(src)
                    src_domain = urlparse(urljoin("http://" + domain, src_str)).netloc
                    if src_domain and src_domain != domain:
                        external_count += 1

            if total_count == 0:
                return 1.0

            ratio = external_count / total_count
            if ratio < 0.22:
                return 1.0
            elif ratio <= 0.61:
                return 0.0
            else:
                return -1.0
        except Exception as e:
            failed_features.append("Request_URL")
            logger.debug(f"提取Request_URL失败: {str(e)}")
            return 0.0

    def _extract_url_of_anchor(
        self, html_content: str, domain: str, failed_features: list
    ) -> float:
        """提取特征14: URL_of_Anchor"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            external_count = 0
            total_count = 0

            for tag in soup.find_all("a"):
                href = tag.get("href")
                if href:
                    href_str = self._to_str(href)
                    if not href_str.startswith(("#", "javascript:", "mailto:")):
                        total_count += 1
                        href_domain = urlparse(
                            urljoin("http://" + domain, href_str)
                        ).netloc
                        if href_domain and href_domain != domain:
                            external_count += 1

            if total_count == 0:
                return 1.0

            ratio = external_count / total_count
            if ratio < 0.31:
                return 1.0
            elif ratio <= 0.67:
                return 0.0
            else:
                return -1.0
        except Exception as e:
            failed_features.append("URL_of_Anchor")
            logger.debug(f"提取URL_of_Anchor失败: {str(e)}")
            return 0.0

    def _extract_links_in_tags(
        self, html_content: str, domain: str, failed_features: list
    ) -> float:
        """提取特征15: Links_in_tags"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            external_count = 0
            total_count = 0

            for tag in soup.find_all(["meta", "script", "link"]):
                src = tag.get("src") or tag.get("href")
                if src:
                    total_count += 1
                    src_str = self._to_str(src)
                    src_domain = urlparse(urljoin("http://" + domain, src_str)).netloc
                    if src_domain and src_domain != domain:
                        external_count += 1

            if total_count == 0:
                return 1.0

            ratio = external_count / total_count
            if ratio < 0.17:
                return 1.0
            elif ratio <= 0.81:
                return 0.0
            else:
                return -1.0
        except Exception as e:
            failed_features.append("Links_in_tags")
            logger.debug(f"提取Links_in_tags失败: {str(e)}")
            return 0.0

    def _extract_sfh(
        self, html_content: str, domain: str, failed_features: list
    ) -> float:
        """提取特征16: SFH"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            forms = soup.find_all("form")
            if not forms:
                return 0.0

            for form in forms:
                action = self._to_str(form.get("action", ""))
                action = action.strip()
                if not action or action == "about:blank":
                    return 0.0
                action_domain = urlparse(urljoin("http://" + domain, action)).netloc
                if action_domain == domain or not action_domain:
                    return 1.0
                else:
                    return -1.0

            return 1.0
        except Exception as e:
            failed_features.append("SFH")
            logger.debug(f"提取SFH失败: {str(e)}")
            return 0.0

    def _extract_submitting_to_email(
        self, html_content: str, failed_features: list
    ) -> float:
        """提取特征17: Submitting_to_email"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            forms = soup.find_all("form")
            for form in forms:
                action = self._to_str(form.get("action", ""))
                action = action.strip()
                if "mailto:" in action.lower():
                    return -1.0
            return 1.0
        except Exception as e:
            failed_features.append("Submitting_to_email")
            logger.debug(f"提取Submitting_to_email失败: {str(e)}")
            return 0.0

    def _extract_abnormal_url(
        self, url: str, domain: str, failed_features: list
    ) -> float:
        """提取特征18: Abnormal_URL"""
        try:
            # 提取纯域名（去掉www等前缀）
            clean_domain = self._extract_clean_domain(domain)
            if not clean_domain:
                failed_features.append("Abnormal_URL")
                return 0.0

            whois_info = self._get_whois_info(clean_domain)
            if not whois_info:
                failed_features.append("Abnormal_URL")
                return 0.0

            # 获取WHOIS中的域名
            whois_domain = whois_info.get("domain_name")
            if whois_domain:
                # 处理可能是列表的情况
                if isinstance(whois_domain, list):
                    whois_domain = whois_domain[0]
                if isinstance(whois_domain, str):
                    whois_domain = whois_domain.lower().strip()
                    clean_domain_lower = clean_domain.lower().strip()
                    # 比较域名是否一致
                    if (
                        clean_domain_lower == whois_domain
                        or clean_domain_lower in whois_domain
                        or whois_domain in clean_domain_lower
                    ):
                        return 1.0
                    else:
                        return -1.0

            # 如果无法比较，默认返回正常
            return 1.0
        except Exception as e:
            failed_features.append("Abnormal_URL")
            logger.debug(f"提取Abnormal_URL失败: {str(e)}")
            return 0.0

    def _extract_redirect(self, url: str, failed_features: list) -> float:
        """提取特征19: Redirect"""
        try:
            response = self.session.get(url, allow_redirects=True, timeout=self.timeout)
            redirect_count = len(response.history)
            if redirect_count <= 1:
                return 0.0
            else:
                return 1.0
        except Exception as e:
            failed_features.append("Redirect")
            logger.debug(f"提取Redirect失败: {str(e)}")
            return 0.0

    def _extract_on_mouseover(self, html_content: str, failed_features: list) -> float:
        """提取特征20: on_mouseover"""
        try:
            if (
                "onmouseover" in html_content.lower()
                and "window.status" in html_content.lower()
            ):
                return -1.0
            return 1.0
        except Exception as e:
            failed_features.append("on_mouseover")
            logger.debug(f"提取on_mouseover失败: {str(e)}")
            return 0.0

    def _extract_right_click(self, html_content: str, failed_features: list) -> float:
        """提取特征21: RightClick"""
        try:
            patterns = [
                "event.button==2",
                "oncontextmenu",
                "addEventListener.*contextmenu",
            ]
            html_lower = html_content.lower()
            for pattern in patterns:
                if pattern in html_lower:
                    return -1.0
            return 1.0
        except Exception as e:
            failed_features.append("RightClick")
            logger.debug(f"提取RightClick失败: {str(e)}")
            return 0.0

    def _extract_popup_window(self, html_content: str, failed_features: list) -> float:
        """提取特征22: popUpWidnow"""
        try:
            patterns = ["window.open", "prompt(", "alert("]
            html_lower = html_content.lower()
            for pattern in patterns:
                if pattern in html_lower:
                    return -1.0
            return 1.0
        except Exception as e:
            failed_features.append("popUpWidnow")
            logger.debug(f"提取popUpWidnow失败: {str(e)}")
            return 0.0

    def _extract_iframe(self, html_content: str, failed_features: list) -> float:
        """提取特征23: Iframe"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            iframes = soup.find_all("iframe")
            if not iframes:
                return 1.0

            for iframe in iframes:
                width = iframe.get("width", "")
                height = iframe.get("height", "")
                if width == "0" or height == "0" or iframe.get("frameborder") == "0":
                    return -1.0
            return 1.0
        except Exception as e:
            failed_features.append("Iframe")
            logger.debug(f"提取Iframe失败: {str(e)}")
            return 0.0

    def _extract_age_of_domain(self, domain: str, failed_features: list) -> float:
        """提取特征24: age_of_domain"""
        try:
            # 提取纯域名（去掉www等前缀）
            clean_domain = self._extract_clean_domain(domain)
            if not clean_domain:
                failed_features.append("age_of_domain")
                return 0.0

            whois_info = self._get_whois_info(clean_domain)
            if not whois_info:
                failed_features.append("age_of_domain")
                return 0.0

            # 获取创建日期
            creation_date = whois_info.get("creation_date")
            if creation_date:
                # 处理可能是列表的情况
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]

                if isinstance(creation_date, datetime):
                    # 计算域名年龄（月）
                    age_months = (datetime.now() - creation_date).days / 30.0
                    # 域名 ≥ 6个月返回1，否则返回-1
                    return 1.0 if age_months >= 6.0 else -1.0

            failed_features.append("age_of_domain")
            return 0.0
        except Exception as e:
            failed_features.append("age_of_domain")
            logger.debug(f"提取age_of_domain失败: {str(e)}")
            return 0.0

    def _extract_dns_record(self, domain: str, failed_features: list) -> float:
        """提取特征25: DNSRecord"""
        try:
            socket.gethostbyname(domain)
            return 1.0
        except Exception as e:
            failed_features.append("DNSRecord")
            logger.debug(f"提取DNSRecord失败: {str(e)}")
            return 0.0

    def _extract_web_traffic(self, domain: str, failed_features: list) -> float:
        """提取特征26: web_traffic"""
        try:
            # 需要Alexa API或其他流量排名服务，这里简化处理
            failed_features.append("web_traffic")
            return 0.0
        except Exception as e:
            failed_features.append("web_traffic")
            logger.debug(f"提取web_traffic失败: {str(e)}")
            return 0.0

    def _extract_page_rank(self, domain: str, failed_features: list) -> float:
        """提取特征27: Page_Rank"""
        try:
            # Google PageRank已停止公开，需要替代服务，这里简化处理
            failed_features.append("Page_Rank")
            return 0.0
        except Exception as e:
            failed_features.append("Page_Rank")
            logger.debug(f"提取Page_Rank失败: {str(e)}")
            return 0.0

    def _extract_google_index(self, domain: str, failed_features: list) -> float:
        """提取特征28: Google_Index"""
        try:
            # 需要Google Search API，这里简化处理
            failed_features.append("Google_Index")
            return 0.0
        except Exception as e:
            failed_features.append("Google_Index")
            logger.debug(f"提取Google_Index失败: {str(e)}")
            return 0.0

    def _extract_links_pointing_to_page(
        self, domain: str, failed_features: list
    ) -> float:
        """提取特征29: Links_pointing_to_page"""
        try:
            # 需要SEO API服务，这里简化处理
            failed_features.append("Links_pointing_to_page")
            return 0.0
        except Exception as e:
            failed_features.append("Links_pointing_to_page")
            logger.debug(f"提取Links_pointing_to_page失败: {str(e)}")
            return 0.0

    def _extract_statistical_report(
        self, url: str, domain: str, failed_features: list
    ) -> float:
        """提取特征30: Statistical_report"""
        try:
            # 需要PhishTank API或Google Safe Browsing API，这里简化处理
            failed_features.append("Statistical_report")
            return 0.0
        except Exception as e:
            failed_features.append("Statistical_report")
            logger.debug(f"提取Statistical_report失败: {str(e)}")
            return 0.0

    def _extract_clean_domain(self, domain: str) -> Optional[str]:
        """
        提取纯域名（去掉www等前缀和端口号）

        Args:
            domain: 原始域名

        Returns:
            清理后的域名，如果无法提取则返回None
        """
        try:
            if not domain:
                return None

            # 去掉端口号
            domain = domain.split(":")[0]

            # 去掉www等常见前缀
            domain = domain.lower().strip()
            for prefix in self.domain_prefixes:
                if domain.startswith(prefix):
                    domain = domain[len(prefix) :]
                    break

            return domain if domain else None
        except Exception as e:
            logger.debug(f"提取纯域名失败: {domain}, 错误: {str(e)}")
            return None

    def _get_whois_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        获取WHOIS信息（带缓存）

        Args:
            domain: 域名

        Returns:
            WHOIS信息字典，如果查询失败则返回None
        """
        try:
            # 检查whois库是否可用
            if whois is None:
                logger.debug("python-whois库未安装，无法查询WHOIS信息")
                return None

            # 检查缓存
            if domain in self._whois_cache:
                return self._whois_cache[domain]

            # 查询WHOIS
            logger.debug(f"查询WHOIS信息: {domain}")
            whois_info = whois.whois(domain)

            # 如果查询失败，whois_info可能是None或空字典
            if not whois_info or (
                isinstance(whois_info, dict) and not whois_info.get("domain_name")
            ):
                logger.debug(f"WHOIS查询失败或域名不存在: {domain}")
                self._whois_cache[domain] = None
                return None

            # 转换为字典格式（如果whois返回的是对象）
            if hasattr(whois_info, "__dict__"):
                whois_dict = whois_info.__dict__
            elif isinstance(whois_info, dict):
                whois_dict = whois_info
            else:
                whois_dict = {}

            # 缓存结果
            self._whois_cache[domain] = whois_dict
            logger.debug(f"WHOIS查询成功: {domain}")
            return whois_dict

        except Exception as e:
            # 处理各种WHOIS查询异常
            error_msg = str(e)
            if (
                "No match" in error_msg
                or "not found" in error_msg.lower()
                or "no entries found" in error_msg.lower()
            ):
                logger.debug(
                    f"WHOIS查询错误（域名可能不存在）: {domain}, 错误: {error_msg}"
                )
            else:
                logger.debug(f"WHOIS查询异常: {domain}, 错误: {error_msg}")
            self._whois_cache[domain] = None
            return None

    def _fetch_html(self, url: str) -> Optional[str]:
        """获取HTML内容"""
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.debug(f"获取HTML内容失败: {url}, 错误: {str(e)}")
            return None
