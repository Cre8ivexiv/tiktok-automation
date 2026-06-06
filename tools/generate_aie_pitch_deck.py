from __future__ import annotations

import math
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pywintypes
import win32com.client as win32
from win32com.client import constants as win32_constants


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "aie_pitch_deck"
ASSET_DIR = OUTPUT_DIR / "assets"
PREVIEW_DIR = OUTPUT_DIR / "preview"
PPT_PATH = OUTPUT_DIR / "AIE_Enterprise_Practice_Pitch_Deck.pptx"
GUIDE_PATH = OUTPUT_DIR / "AIE_Enterprise_Practice_Pitch_Deck_Guide.md"

SLIDE_W = 960
SLIDE_H = 540

FONT_NAME = "Segoe UI"
PP_ALIGN_LEFT = 1
PP_ALIGN_CENTER = 2
PP_LAYOUT_BLANK = 12
MSO_TEXT_ORIENTATION_HORIZONTAL = 1
PP_SLIDE_SIZE_16X9 = 16


class ConstantProxy:
    def __init__(self, base):
        self.base = base
        self.fallback = {
            "ppAlignLeft": 1,
            "ppAlignCenter": 2,
            "ppAlignRight": 3,
            "ppAdvanceOnClick": 1,
            "ppEffectFade": 1793,
            "ppEffectAppear": 3844,
            "ppEffectWipeRight": 2819,
            "msoShapeRectangle": 1,
            "msoShapeRoundedRectangle": 5,
            "msoShapeOval": 9,
            "msoShapeArc": 25,
            "msoShapeChevron": 52,
            "msoAnchorMiddle": 3,
        }

    def __getattr__(self, name: str):
        try:
            return getattr(self.base, name)
        except AttributeError:
            if name in self.fallback:
                return self.fallback[name]
            raise


constants = ConstantProxy(win32_constants)


def retry_com(action, attempts: int = 10, delay: float = 0.08):
    for attempt in range(attempts):
        try:
            return action()
        except pywintypes.com_error as exc:
            if "Call was rejected by callee" not in str(exc) or attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def set_shape_text(
    shape,
    text: str,
    size: int,
    color: tuple[int, int, int] | None = None,
    bold: bool = False,
    align: int = PP_ALIGN_LEFT,
    font_name: str = FONT_NAME,
    margin_left: int | None = None,
    margin_right: int | None = None,
    margin_top: int | None = None,
    margin_bottom: int | None = None,
    vertical_anchor: int | None = None,
):
    if color is None:
        color = WHITE

    def configure():
        shape.TextFrame.TextRange.Text = text
        shape.TextFrame.TextRange.Font.Name = font_name
        shape.TextFrame.TextRange.Font.Size = size
        shape.TextFrame.TextRange.Font.Bold = -1 if bold else 0
        shape.TextFrame.TextRange.Font.Color.RGB = rgb(*color)
        shape.TextFrame.TextRange.ParagraphFormat.Alignment = align
        if margin_left is not None:
            shape.TextFrame.MarginLeft = margin_left
        if margin_right is not None:
            shape.TextFrame.MarginRight = margin_right
        if margin_top is not None:
            shape.TextFrame.MarginTop = margin_top
        if margin_bottom is not None:
            shape.TextFrame.MarginBottom = margin_bottom
        if vertical_anchor is not None:
            shape.TextFrame.VerticalAnchor = vertical_anchor

    retry_com(configure)
    return shape


def rgb(r: int, g: int, b: int) -> int:
    return r + (g * 256) + (b * 65536)


BG = (5, 10, 20)
BG_ALT = (8, 16, 30)
CARD = (14, 24, 44)
CARD_ALT = (10, 20, 36)
CARD_BORDER = (46, 93, 155)
ACCENT = (0, 183, 255)
ACCENT_SOFT = (71, 140, 255)
WHITE = (244, 248, 255)
MUTED = (165, 184, 212)
MUTED_2 = (118, 137, 165)
WARNING = (255, 107, 74)
WARNING_SOFT = (255, 152, 89)
SUCCESS = (58, 214, 151)
GRID = (26, 43, 71)


@dataclass
class SlideSpec:
    number: int
    title: str
    content_lines: list[str]
    visual: str
    speaker_notes: str
    animation: str
    script_segment: str
    layout_key: str


SOURCES = [
    (
        "Stack Overflow Developer Survey 2025 - AI",
        "https://survey.stackoverflow.co/2025/ai",
        "Verified: 84% use or plan to use AI tools; 51% of professional developers use AI tools daily; 46% actively distrust AI accuracy.",
    ),
    (
        "Veracode 2025 GenAI Code Security Report",
        "https://www.veracode.com/blog/genai-code-security-report/",
        "Verified: 45% of code samples failed security tests, implying 55% passed.",
    ),
    (
        "IBM Cost of a Data Breach Report 2024 insight article",
        "https://www.ibm.com/think/insights/whats-new-2024-cost-of-a-data-breach-report",
        "Verified: the 2024 global average breach cost reached USD 4.88 million.",
    ),
    (
        "Verizon 2025 Data Breach Investigations Report press release",
        "https://www.verizon.com/about/news/2025-data-breach-investigations-report",
        "Verified: exploitation of vulnerabilities surged by 34% in the 2025 DBIR.",
    ),
    (
        "Verizon article citing the 2025 DBIR",
        "https://www.verizon.com/business/resources/articles/s/how-to-protect-your-organization-from-a-pretexting-attack/",
        "Verified: 60% of breaches analyzed in Verizon's 2025 DBIR involved some kind of human element.",
    ),
    (
        "Palo Alto Networks Unit 42 Global Incident Response Report press release",
        "https://investors.paloaltonetworks.com/news-releases/news-release-details/unit-42-report-ai-and-attack-surface-complexity-fuel-majority",
        "Verified on February 17, 2026: 90% of breaches linked to misconfigurations or security gaps; 87% spanned multiple attack surfaces.",
    ),
    (
        "ISC2 2024 Cybersecurity Workforce Study",
        "https://www.isc2.org/insights/2024/10/isc2-2024-cybersecurity-workforce-study",
        "Verified: workforce gap estimated at 4,763,963 people.",
    ),
    (
        "Gartner security spending forecast press release",
        "https://www.gartner.com/en/newsroom/press-releases/2025-07-29-gartner-forecasts-worldwide-end-user-spending-on-information-security-to-total-213-billion-us-dollars-in-2025",
        "Verified on July 29, 2025: spending projected at $193B in 2024, $213B in 2025, and $240B in 2026.",
    ),
    (
        "Cybersecurity Ventures cybercrime damage forecast",
        "https://cybersecurityventures.com/official-cybercrime-report-2025/",
        "Supporting market signal used for the $10.5T annual cybercrime cost narrative.",
    ),
    (
        "OWASP Top 10",
        "https://owasp.org/Top10/",
        "Used to frame common application security risks addressed by AIE.",
    ),
    (
        "Snyk product page",
        "https://snyk.io/product/",
        "Used for competitor positioning.",
    ),
    (
        "Thinkst Canary product page",
        "https://canary.tools/",
        "Used for competitor positioning.",
    ),
    (
        "Palo Alto Prisma Cloud product page",
        "https://www.paloaltonetworks.com/prisma/cloud",
        "Used for competitor positioning.",
    ),
    (
        "Tenable products",
        "https://www.tenable.com/products",
        "Used for competitor positioning.",
    ),
    (
        "AttackIQ platform page",
        "https://www.attackiq.com/",
        "Used for competitor positioning.",
    ),
]


SLIDES: list[SlideSpec] = [
    SlideSpec(
        1,
        "AIE: Autonomous Security Engineer",
        [
            "Security can't stay manual. It has to be autonomous.",
            "Ibrahim Timilehin",
            "BCU Enterprise Practice Project",
            "AIE = Artificial Intelligence Engineer",
            "Autonomous security for the vibe coding era",
        ],
        "Futuristic cloud-security hero visual with a shield, cloud outline, and circuit pattern.",
        "Good afternoon, my name is Ibrahim Timilehin, and today I'm pitching AIE, which stands for Artificial Intelligence Engineer. AIE is an autonomous security engineer in the cloud, built for the way software is being developed today.",
        "Fade in title first, then subtitle, then presenter details.",
        "Good afternoon, my name is Ibrahim Timilehin, and today I'm pitching AIE, which stands for Artificial Intelligence Engineer. AIE is an autonomous security engineer in the cloud, built for the way software is being developed today.",
        "title",
    ),
    SlideSpec(
        2,
        "The Era of Vibe Coding",
        [
            "How many of you here code with AI?",
            "AI helps developers build faster, but faster does not always mean safer.",
            "84% of developers use or plan to use AI tools.",
            "51% of professional developers use AI tools daily.",
            "46% of developers distrust AI output.",
            "Source: Stack Overflow Developer Survey 2025",
        ],
        "Split-screen visual: abstract code-assistant workspace on one side and a security warning panel on the other, with three stat cards.",
        "Before I explain the product, I want to ask a quick question. How many of you here have used AI when coding, what people now call vibe coding? Most of us either have, or we know someone who does. AI can generate code, explain errors, and speed up development, but adoption is moving faster than trust and security.",
        "Reveal the question first, then the supporting line, then show the three stats one by one.",
        "Before I explain the product, I want to ask a quick question. How many of you here have used AI when coding, what people now call vibe coding? Most of us either have, or we know someone who does. Stack Overflow's 2025 survey shows adoption is high, but trust is still a real issue.",
        "hook",
    ),
    SlideSpec(
        3,
        "AI Helps Us Build Faster - Not Always Safer",
        [
            "AI-generated and human-written code can both introduce hidden vulnerabilities.",
            "55% of AI-generated code was secure.",
            "45% failed security tests.",
            "60% of breaches still involve the human element.",
            "Vulnerability exploitation increased by 34%.",
        ],
        "Bar or donut chart for secure vs vulnerable AI-generated code, plus warning chips for insecure auth, SQL injection, exposed secrets, weak cryptography, and cloud misconfiguration.",
        "Here is the problem. AI does not just help us code faster. It can also make us insecure faster. Veracode found that only 55% of AI-generated code was secure, which means around 45% failed security tests. But this is not only an AI problem. Developers still create weak authentication, exposed secrets, poor cryptography, and cloud misconfigurations.",
        "Animate the chart first, then reveal the risk chips and supporting breach metrics.",
        "That speed creates a security problem. Veracode found that 45% of AI-generated code samples failed security tests, and Verizon still links many breaches to human error and vulnerability exploitation. So the risk is not just bad code, it is fast-moving insecure delivery.",
        "problem",
    ),
    SlideSpec(
        4,
        "Who Is Affected?",
        [
            "Developers using AI-generated code and fast workflows",
            "DevOps teams managing deployment, cloud access, and infrastructure",
            "Cloud-first startups and SMEs without large security teams",
            "Enterprise software teams with compliance needs",
            "Security teams overloaded by alerts and manual checks",
            "Any organisation shipping software faster than it can secure it",
        ],
        "Five stakeholder cards with clean badge icons for Developer, DevOps, Startup/SME, Enterprise, and Security Team.",
        "This problem affects more than just security teams. Developers are affected because vulnerable code can be shipped without being noticed. DevOps teams are affected because cloud infrastructure can be misconfigured. Startups, SMEs, and enterprises are all affected because they are expected to deliver quickly while staying compliant and resilient.",
        "Show each stakeholder card one by one, then reveal the closing statement.",
        "This affects developers, DevOps teams, startups, SMEs, enterprise teams, and overloaded security functions. In short, it affects any organisation shipping software faster than it can secure it.",
        "affected",
    ),
    SlideSpec(
        5,
        "Scaling Vulnerabilities at Speed",
        [
            "Organisations are not just facing more attacks. They are facing faster attacks.",
            "$4.88M average global cost of a data breach in 2024.",
            "$10.5T predicted annual cybercrime cost by 2025.",
            "90% of breaches linked to misconfigurations or security gaps.",
            "87% of incidents span multiple attack surfaces.",
            "4,763,963 global cybersecurity workforce gap.",
        ],
        "Four large metric cards plus a supporting strip for multi-surface complexity.",
        "This matters because small flaws do not stay small. IBM reported the average data breach cost reached 4.88 million dollars in 2024. Cybercrime is predicted to cost the world 10.5 trillion dollars annually by 2025. Palo Alto links most breaches to misconfigurations or security gaps, and ISC2 estimates the workforce gap at more than 4.7 million people.",
        "Make the metric cards appear sequentially with a subtle fade or slight zoom.",
        "The timing matters because the cost of failure is high and the security workforce is stretched. Attacks are faster, environments are more complex, and organisations do not have enough people to check everything manually.",
        "why_now",
    ),
    SlideSpec(
        6,
        "AIE Closes the Gap",
        [
            "AIE is an autonomous security platform that continuously monitors, learns, validates, and helps fix risks across code and cloud.",
            "Detect -> Learn -> Simulate -> Fix -> Improve",
            "Focus areas: secure code, secure cloud, learn from attackers, validate defences",
        ],
        "Circular or loop-style diagram with AIE in the centre and a five-step adaptive cycle around it.",
        "That is the gap AIE is designed to close. AIE is not just another alert dashboard. It is an autonomous security layer that monitors code, checks cloud infrastructure, learns from attacker behaviour, and validates whether defences work. The key idea is the loop: detect, learn, simulate, fix, and improve.",
        "Animate the loop clockwise so the model feels continuous rather than static.",
        "AIE closes that gap by acting as an autonomous security layer across both code and cloud. Its value is the loop: detect, learn, simulate, fix, and improve.",
        "solution",
    ),
    SlideSpec(
        7,
        "One Platform, Four Core Layers",
        [
            "1. Secure Coding Assistant",
            "2. Cloud Misconfiguration Scanner",
            "3. Honeypot Intelligence Layer",
            "4. Red vs Blue Simulation Engine",
            "Anchored in OWASP risks such as injection, cryptographic failures, security misconfiguration, and authentication failures.",
        ],
        "Four-quadrant product map with clean cards for code, cloud, honeypot, and simulation.",
        "AIE has four core layers. The first is a secure coding assistant that checks code as developers write it. The second is a cloud scanner that looks for risky permissions and posture gaps. The third is honeypot intelligence, where AIE learns from real attacker behaviour. The fourth is a simulation engine that validates whether defences actually work.",
        "Fade in each quadrant one at a time.",
        "The platform has four layers: a secure coding assistant, a cloud scanner, a honeypot intelligence layer, and a simulation engine. Together they move AIE beyond detection into adaptive defence.",
        "layers",
    ),
    SlideSpec(
        8,
        "Autonomous Adaptive Loop",
        [
            "1. Detect: scan code and cloud for vulnerabilities.",
            "2. Learn: capture active exploit patterns through honeypots.",
            "3. Simulate: validate resilience with attack testing.",
            "4. Fix: suggest or apply remediation.",
            "5. Improve: push new intelligence back into detection.",
            "AIE becomes more useful over time because each layer teaches the others.",
        ],
        "Architecture flow from developer and AI-generated code through scanner, cloud checks, honeypot, simulation, and remediation.",
        "The important part is how these features connect. AIE detects issues in code and cloud environments. It learns from honeypots. It runs simulations to test defences. Then it recommends or applies fixes. Over time, if one layer learns something new, that learning strengthens the others.",
        "Use a left-to-right flow animation with the final loop-back appearing last.",
        "The key is the adaptive loop. AIE detects, learns, simulates, and fixes, then feeds that intelligence back into the platform so it improves over time.",
        "how_it_works",
    ),
    SlideSpec(
        9,
        "Strategic Business Benefits",
        [
            "Reduce risk and cost",
            "Operational speed",
            "Customer trust",
            "Long-term resilience",
            "AI and automation can reduce breach costs when used properly.",
        ],
        "Four benefit cards with strong icon badges for Risk, Speed, Trust, and Resilience.",
        "The benefit to organisations is clear. AIE helps catch vulnerabilities earlier, reduce breach risk, reduce manual workload, and improve customer trust. This supports business goals because companies want to innovate quickly, but they cannot afford to let security become an afterthought.",
        "Bring in the benefit cards one by one.",
        "Business value is straightforward: reduce risk, support faster delivery, strengthen trust, and turn real attack intelligence into long-term resilience.",
        "benefits",
    ),
    SlideSpec(
        10,
        "The Adaptive Advantage",
        [
            "Existing platforms focus on individual silos; AIE integrates them into one adaptive system.",
            "Snyk: strong on code security",
            "Thinkst Canary: strong on deception",
            "Prisma Cloud and Tenable: strong on posture and exposure",
            "AttackIQ: strong on validation",
            "AIE advantage: connect code, cloud, honeypots, and simulation in one loop",
        ],
        "Modern comparison table with the AIE advantage column highlighted in blue.",
        "Existing tools are useful, but they mostly solve one part of the problem. Snyk focuses on code. Thinkst Canary focuses on deception. Prisma Cloud and Tenable focus on cloud posture and exposure management. AttackIQ focuses on validation. AIE's advantage is that it brings the layers together so learning in one area improves the others.",
        "Reveal competitor rows first, then highlight the AIE advantage column last.",
        "What makes AIE different is integration. Instead of treating code security, cloud posture, deception, and validation as separate products, AIE connects them into one adaptive system.",
        "competition",
    ),
    SlideSpec(
        11,
        "SaaS Subscription Model",
        [
            "Starter - GBP29/month",
            "Growth - GBP499/month",
            "Enterprise - GBP2,500+/month",
            "Priced by team size, repositories, cloud assets, and infrastructure footprint.",
            "Security spending forecast: $193B in 2024, $213B in 2025, $240B in 2026.",
        ],
        "Three pricing cards with a compact line chart for security spending growth.",
        "AIE follows a subscription model. Small teams can start with basic scanning while larger organisations can pay for automation, simulation, compliance support, and deeper integrations. Gartner projects security spending to continue rising from 193 billion dollars in 2024 to 240 billion dollars in 2026.",
        "Slide the pricing cards up one by one and draw or wipe in the line chart.",
        "Commercially, AIE works as a SaaS subscription. Teams can start small, expand into cloud posture and honeypots, and scale into automation and simulation as their environments grow.",
        "business_model",
    ),
    SlideSpec(
        12,
        "Year One Investment Plan",
        [
            "Investment need: GBP100,000 - GBP150,000",
            "Funding sources: founder contribution, BCU innovation support, startup grants, seed investment, pilot partner support",
            "Cost allocation: 55% product and security research; 18% cloud and testing; 10% legal and compliance; 12% marketing and sales; 5% support and pilot testing",
        ],
        "Two-column layout with a funding summary card and a donut chart using the exact percentage split.",
        "For year one, the estimated investment need is between one hundred thousand and one hundred and fifty thousand pounds. This supports product development, security research, cloud testing, legal compliance, marketing, and pilot support. Funding can come from founder contribution, BCU support, grants, seed investment, and pilot partners.",
        "Reveal the funding need first, then bring in the donut chart and legend.",
        "Year one requires focused early investment to build the MVP, validate the platform safely in the cloud, and get pilot evidence for the next funding stage.",
        "investment",
    ),
    SlideSpec(
        13,
        "Why I Can Build This",
        [
            "Cybersecurity background from my degree and practical security modules",
            "Knowledge of cloud security and misconfiguration risk",
            "Understanding of secure coding and OWASP risks",
            "Experience with security operations, incident response, and risk management",
            "Strong interest in automation, AI security, and developer-focused tools",
            "Built through my BCU Enterprise Practice Project and innovation journey",
        ],
        "Skill icons or capability cards on the left with a roadmap from Learning -> MVP -> Pilot -> Adaptive Platform on the right.",
        "This project is not random for me. It connects directly to what I have been learning in cybersecurity, especially cloud security, secure coding, vulnerability management, security operations, and incident response. It also supports my employability because it shows I can turn technical learning into a practical business idea.",
        "Fade in the capability cards, then wipe the roadmap from left to right.",
        "This project fits my background because it connects cybersecurity learning with cloud security, secure coding, incident response, and automation. It also shows how I can turn technical knowledge into an industry-facing solution.",
        "team",
    ),
    SlideSpec(
        14,
        "From Scanner to Autonomous Platform",
        [
            "The idea started as a security scanner.",
            "Feedback showed a scanner alone would not be innovative enough.",
            "The concept improved by adding honeypots, cloud remediation, and red-vs-blue simulation.",
            "The model shifted to subscription SaaS because security requires continuous protection.",
            "Feedback helped turn AIE from a tool into a platform.",
        ],
        "Before-and-after diagram contrasting a simple scanner with an adaptive platform.",
        "The idea has developed through feedback. At first, it could have been just a scanner, but that would not be strong enough because many scanners already exist. Feedback helped improve the idea by adding honeypots, cloud remediation, and simulation, which changed AIE from a one-off tool into a continuous security platform.",
        "Show the before state first, then the arrow, then the after state.",
        "The concept has already improved through feedback. What began as a scanner has evolved into a more defensible platform with learning, remediation, and continuous validation.",
        "traction",
    ),
    SlideSpec(
        15,
        "The ASK",
        [
            "Support a 12-month pilot of AIE with funding, mentorship, and access to a safe test environment.",
            "What I need: pilot funding, cybersecurity mentor feedback, a safe cloud test environment, and pilot users.",
            "Success looks like: a working MVP, tested scanning workflow, honeypot prototype, clear pilot feedback, and evidence for future investment.",
            "Security can't stay manual. It has to be autonomous.",
            "Contact: Ibrahim Timilehin - ibrahim.timilehin@bcu.ac.uk",
        ],
        "Bold final statement with a shield mark, strong closing quote, and two support cards.",
        "My ask is simple. I am asking for support for a 12-month pilot of AIE with funding, mentorship, and access to a safe test environment. The goal is to build a working MVP, test the core workflow, collect feedback, and prepare the project for future development. Security cannot stay manual. It has to be autonomous.",
        "Show the ask headline first, then the support needs, then the closing quote and contact line.",
        "My ask is simple: support a 12-month pilot with funding, mentorship, and a safe cloud test environment. The goal is to prove the MVP, validate the workflow, and prepare AIE for future investment.",
        "ask",
    ),
    SlideSpec(
        16,
        "Data Sources",
        [
            "This pitch is supported by current industry research and cybersecurity reports.",
            "Sources include Stack Overflow, Veracode, IBM, Verizon, Gartner, ISC2, Palo Alto Networks Unit 42, OWASP, and competitor product pages.",
        ],
        "Clean two-column references slide with source names and URLs.",
        "This slide shows that the pitch is supported by recognised industry sources and not just opinion. The data comes from reports by Stack Overflow, Veracode, IBM, Verizon, Gartner, ISC2, Palo Alto, OWASP, and relevant competitor platforms.",
        "No animation, or a simple overall fade if needed.",
        "The pitch is grounded in current industry data, which shows why AIE is timely, credible, and relevant to real-world software security challenges.",
        "sources",
    ),
]


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/seguisb.ttf"),
                Path("C:/Windows/Fonts/segoeuib.ttf"),
                Path("C:/Windows/Fonts/aptos-bold.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/aptos.ttf"),
                Path("C:/Windows/Fonts/calibri.ttf"),
            ]
        )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def rounded_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill, outline=None, radius=28, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    spacing: int = 8,
) -> int:
    wrapped = []
    for paragraph in text.split("\n"):
        if not paragraph:
            wrapped.append("")
            continue
        lines = []
        words = paragraph.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        wrapped.extend(lines)
    draw.multiline_text(xy, "\n".join(wrapped), font=font, fill=fill, spacing=spacing)
    bbox = draw.multiline_textbbox(xy, "\n".join(wrapped), font=font, spacing=spacing)
    return bbox[3] - bbox[1]


def add_glow(img: Image.Image, center: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: int):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = center
    for step in range(radius, 0, -16):
        a = int(alpha * (step / radius) ** 2)
        draw.ellipse((x - step, y - step, x + step, y + step), fill=(*color, a))
    overlay = overlay.filter(ImageFilter.GaussianBlur(38))
    img.alpha_composite(overlay)


def make_background(path: Path, cover: bool = False):
    width, height = 1920, 1080
    img = Image.new("RGBA", (width, height), BG + (255,))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        t = y / max(1, height - 1)
        color = blend(BG, BG_ALT, t)
        draw.line((0, y, width, y), fill=color, width=1)

    add_glow(img, (1440, 240), 340, ACCENT, 120)
    add_glow(img, (420, 860), 260, ACCENT_SOFT, 70)
    if cover:
        add_glow(img, (1460, 520), 440, ACCENT, 90)
        add_glow(img, (1380, 540), 220, WARNING_SOFT, 28)

    draw = ImageDraw.Draw(img)
    for x in range(80, width, 120):
        draw.line((x, 0, x, height), fill=GRID + (46,), width=1)
    for y in range(60, height, 96):
        draw.line((0, y, width, y), fill=GRID + (36,), width=1)

    for idx in range(22):
        base_x = 1120 + idx * 26
        draw.line((base_x, 120, base_x, 940), fill=(60, 110, 180, 54), width=2)
        if idx % 3 == 0:
            draw.ellipse((base_x - 6, 220 + idx * 18 % 500, base_x + 6, 232 + idx * 18 % 500), fill=(72, 214, 255, 110))

    if cover:
        shield = [(1370, 190), (1230, 260), (1230, 510), (1370, 726), (1510, 510), (1510, 260)]
        draw.line(shield + [shield[0]], fill=(*ACCENT, 220), width=9)
        draw.arc((1250, 315, 1490, 535), 205, 335, fill=(*WHITE, 180), width=8)
        draw.arc((1320, 280, 1560, 500), 25, 155, fill=(*WHITE, 180), width=8)
        draw.arc((1360, 340, 1600, 560), 205, 335, fill=(*WHITE, 180), width=8)
        for y in (292, 354, 416, 478):
            draw.line((1170, y, 1320, y), fill=(70, 136, 210, 120), width=4)
            draw.ellipse((1314, y - 7, 1328, y + 7), fill=(72, 214, 255, 150))

    img.save(path)


def make_hook_visual(path: Path):
    img = Image.new("RGBA", (900, 540), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rounded_box(draw, (20, 20, 880, 520), (10, 18, 34, 235), outline=(52, 92, 148, 150), radius=40, width=2)
    rounded_box(draw, (52, 70, 448, 470), (14, 26, 48, 245), outline=(48, 86, 142, 180), radius=28)
    rounded_box(draw, (500, 80, 830, 260), (28, 18, 28, 245), outline=(160, 62, 48, 180), radius=28)
    rounded_box(draw, (500, 290, 830, 470), (14, 24, 42, 245), outline=(48, 86, 142, 180), radius=28)

    mono = pil_font(26, bold=False)
    bold = pil_font(34, bold=True)
    title = pil_font(52, bold=True)
    small = pil_font(22, bold=False)

    code_lines = [
        "prompt> build auth api with ai assistant",
        "assistant> generated login route, token helper",
        "warning> secret hard-coded in config",
        "warning> missing input validation",
        "deploy> build passed",
    ]
    y = 120
    for idx, line in enumerate(code_lines):
        color = WHITE if idx < 2 else WARNING_SOFT
        draw.text((82, y), line, font=mono, fill=color)
        y += 58

    draw.text((548, 116), "FAST BUILD", font=bold, fill=WHITE)
    draw.text((548, 168), "Unsafe output scales risk.", font=small, fill=MUTED)
    triangle = [(658, 190), (740, 190), (699, 116)]
    draw.polygon(triangle, outline=WARNING, fill=(90, 32, 24))
    draw.text((690, 146), "!", font=title, fill=WHITE, anchor="mm")

    draw.text((548, 324), "TRUST GAP", font=bold, fill=WHITE)
    draw.text((548, 376), "Adoption is high.\nVerification still matters.", font=small, fill=MUTED, spacing=10)

    img.save(path)


def make_donut_chart(path: Path, values: Sequence[int], labels: Sequence[str], colors: Sequence[tuple[int, int, int]], title: str):
    img = Image.new("RGBA", (900, 540), BG + (255,))
    draw = ImageDraw.Draw(img)
    rounded_box(draw, (18, 18, 882, 522), CARD + (255,), outline=(*CARD_BORDER, 150), radius=36, width=2)

    center = (260, 280)
    outer = 170
    inner = 86
    total = sum(values)
    start = -90
    for value, color in zip(values, colors):
        end = start + (value / total) * 360
        draw.pieslice((center[0] - outer, center[1] - outer, center[0] + outer, center[1] + outer), start, end, fill=color)
        start = end
    draw.ellipse((center[0] - inner, center[1] - inner, center[0] + inner, center[1] + inner), fill=BG)
    header_font = pil_font(36, bold=True)
    body_font = pil_font(24, bold=False)
    small_font = pil_font(20, bold=False)
    draw.text((58, 48), title, font=header_font, fill=WHITE)
    draw.text((260, 280), f"{values[0]}%", font=pil_font(46, bold=True), fill=WHITE, anchor="mm")
    draw.text((260, 332), labels[0], font=small_font, fill=MUTED, anchor="mm")

    legend_y = 148
    for value, label, color in zip(values, labels, colors):
        draw.rounded_rectangle((520, legend_y, 552, legend_y + 32), radius=8, fill=color)
        draw.text((576, legend_y - 2), label, font=body_font, fill=WHITE)
        draw.text((808, legend_y - 2), f"{value}%", font=body_font, fill=color, anchor="ra")
        legend_y += 84

    img.save(path)


def make_bar_chart(path: Path, title: str, data: Sequence[tuple[str, int, tuple[int, int, int]]], show_axis_max: int = 100):
    img = Image.new("RGBA", (1200, 760), BG + (255,))
    draw = ImageDraw.Draw(img)
    rounded_box(draw, (18, 18, 1182, 742), CARD + (255,), outline=(*CARD_BORDER, 150), radius=36, width=2)
    header_font = pil_font(42, bold=True)
    label_font = pil_font(24, bold=False)
    value_font = pil_font(24, bold=True)
    draw.text((52, 40), title, font=header_font, fill=WHITE)

    left = 270
    top = 150
    right = 1090
    bar_h = 54
    gap = 42
    bottom = top + len(data) * (bar_h + gap)
    for tick in range(0, show_axis_max + 1, 20):
        x = left + int((right - left) * (tick / show_axis_max))
        draw.line((x, top - 36, x, bottom - gap + bar_h + 18), fill=(50, 72, 108), width=2)
        draw.text((x, top - 78), str(tick), font=label_font, fill=MUTED, anchor="mm")

    y = top
    for label, value, color in data:
        draw.text((54, y + 8), label, font=label_font, fill=WHITE)
        draw.rounded_rectangle((left, y, right, y + bar_h), radius=20, fill=(18, 28, 46))
        filled = left + int((right - left) * (value / show_axis_max))
        draw.rounded_rectangle((left, y, filled, y + bar_h), radius=20, fill=color)
        draw.text((filled - 18, y + bar_h / 2), f"{value}", font=value_font, fill=BG, anchor="rm")
        y += bar_h + gap

    img.save(path)


def make_line_chart(path: Path):
    data = [("2024", 193), ("2025", 213), ("2026", 240)]
    img = Image.new("RGBA", (920, 460), BG + (255,))
    draw = ImageDraw.Draw(img)
    rounded_box(draw, (16, 16, 904, 444), CARD + (255,), outline=(*CARD_BORDER, 140), radius=32, width=2)
    header = pil_font(34, bold=True)
    label = pil_font(22, bold=False)
    value = pil_font(24, bold=True)
    draw.text((42, 34), "Security Spending Forecast", font=header, fill=WHITE)

    chart_left, chart_top, chart_right, chart_bottom = 90, 110, 820, 360
    min_val, max_val = 180, 250
    for tick in (180, 200, 220, 240):
        y = chart_bottom - int((tick - min_val) / (max_val - min_val) * (chart_bottom - chart_top))
        draw.line((chart_left, y, chart_right, y), fill=(42, 62, 96), width=2)
        draw.text((52, y), f"${tick}B", font=label, fill=MUTED, anchor="rm")

    points = []
    step = (chart_right - chart_left) / (len(data) - 1)
    for idx, (year, amount) in enumerate(data):
        x = int(chart_left + idx * step)
        y = chart_bottom - int((amount - min_val) / (max_val - min_val) * (chart_bottom - chart_top))
        points.append((x, y))
        draw.text((x, chart_bottom + 28), year, font=label, fill=MUTED, anchor="mm")

    for idx in range(len(points) - 1):
        draw.line((points[idx], points[idx + 1]), fill=ACCENT, width=8)
    for (x, y), (_, amount) in zip(points, data):
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=WHITE)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=ACCENT)
        draw.text((x, y - 30), str(amount), font=value, fill=WHITE, anchor="ms")

    img.save(path)


def add_animation(shape, effect: int):
    try:
        retry_com(lambda: setattr(shape.AnimationSettings, "Animate", True))
        retry_com(lambda: setattr(shape.AnimationSettings, "EntryEffect", effect))
    except Exception:
        return


def set_slide_transition(slide):
    try:
        retry_com(lambda: setattr(slide.SlideShowTransition, "EntryEffect", constants.ppEffectFade))
    except Exception:
        return


def style_shape(shape, fill_color: tuple[int, int, int], line_color: tuple[int, int, int] | None = None, transparency: float = 0.0):
    def configure():
        shape.Fill.Visible = True
        shape.Fill.ForeColor.RGB = rgb(*fill_color)
        shape.Fill.Transparency = transparency
        if line_color:
            shape.Line.Visible = True
            shape.Line.ForeColor.RGB = rgb(*line_color)
            shape.Line.Transparency = max(0, min(0.75, transparency))
            shape.Line.Weight = 1.5
        else:
            shape.Line.Visible = False

    retry_com(configure)


def add_textbox(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str,
    size: int,
    color: tuple[int, int, int] = WHITE,
    bold: bool = False,
    align: int = PP_ALIGN_LEFT,
    font_name: str = FONT_NAME,
):
    shape = retry_com(lambda: slide.Shapes.AddTextbox(MSO_TEXT_ORIENTATION_HORIZONTAL, left, top, width, height))
    set_shape_text(
        shape,
        text,
        size,
        color=color,
        bold=bold,
        align=align,
        font_name=font_name,
        margin_left=8,
        margin_right=8,
        margin_top=4,
        margin_bottom=4,
    )
    return shape


def add_card(slide, left, top, width, height, title, body, accent_color=ACCENT, body_size=18):
    card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, top, width, height)
    style_shape(card, CARD, CARD_BORDER, transparency=0.06)
    title_shape = add_textbox(slide, left + 18, top + 14, width - 36, 30, title, 16, color=accent_color, bold=True)
    body_shape = add_textbox(slide, left + 18, top + 42, width - 36, height - 56, body, body_size, color=WHITE)
    return card, title_shape, body_shape


def add_metric_card(slide, left, top, width, height, metric, label, color=ACCENT):
    card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, top, width, height)
    style_shape(card, CARD, CARD_BORDER, transparency=0.03)
    stripe = slide.Shapes.AddShape(constants.msoShapeRectangle, left, top, width, 7)
    style_shape(stripe, color)
    metric_shape = add_textbox(slide, left + 18, top + 20, width - 36, 46, metric, 28, color=WHITE, bold=True)
    label_shape = add_textbox(slide, left + 18, top + 68, width - 36, height - 76, label, 15, color=MUTED)
    return [card, stripe, metric_shape, label_shape]


def add_badge_card(slide, left, top, width, height, badge, title, body):
    card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, top, width, height)
    style_shape(card, CARD, CARD_BORDER, transparency=0.05)
    badge_circle = slide.Shapes.AddShape(constants.msoShapeOval, left + 16, top + 16, 42, 42)
    style_shape(badge_circle, ACCENT)
    badge_text = add_textbox(slide, left + 16, top + 17, 42, 32, badge, 16, color=BG, bold=True, align=constants.ppAlignCenter)
    title_shape = add_textbox(slide, left + 70, top + 14, width - 88, 24, title, 16, color=WHITE, bold=True)
    body_shape = add_textbox(slide, left + 16, top + 62, width - 32, height - 72, body, 15, color=MUTED)
    return [card, badge_circle, badge_text, title_shape, body_shape]


def add_footer(slide, number: int):
    line = slide.Shapes.AddShape(constants.msoShapeRectangle, 0, SLIDE_H - 18, SLIDE_W, 1.4)
    style_shape(line, CARD_BORDER)
    brand = add_textbox(slide, 40, SLIDE_H - 16, 260, 14, "BCU Enterprise Practice Project", 10, color=MUTED)
    num = add_textbox(slide, SLIDE_W - 80, SLIDE_H - 16, 40, 14, f"{number:02d}", 10, color=MUTED, bold=True, align=constants.ppAlignRight)
    return [line, brand, num]


def add_title_block(slide, title: str, section: str | None = None):
    if section:
        pill = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 42, 26, 210, 22)
        style_shape(pill, CARD, ACCENT, transparency=0.0)
        set_shape_text(pill, section, 10, color=ACCENT, bold=True, align=constants.ppAlignCenter, margin_left=0, margin_right=0, margin_top=2, margin_bottom=0)
    title_shape = add_textbox(slide, 42, 44, 620, 46, title, 28, color=WHITE, bold=True)
    rule = slide.Shapes.AddShape(constants.msoShapeRectangle, 42, 92, 108, 4)
    style_shape(rule, ACCENT)
    return [title_shape, rule]


def add_notes(slide, spec: SlideSpec):
    note_text = (
        f"Speaker notes:\r{spec.speaker_notes}\r\r"
        f"Suggested animation:\r{spec.animation}\r\r"
        f"Presenter script segment:\r{spec.script_segment}"
    )
    retry_com(lambda: setattr(slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange, "Text", note_text))


def add_background(slide, asset: Path):
    retry_com(lambda: slide.Shapes.AddPicture(str(asset.resolve()), False, True, 0, 0, SLIDE_W, SLIDE_H))


def add_picture(slide, asset: Path, left: float, top: float, width: float, height: float):
    return retry_com(lambda: slide.Shapes.AddPicture(str(asset.resolve()), False, True, left, top, width, height))


def make_table_cell(slide, left, top, width, height, text, fill, color=WHITE, bold=False, size=14, align=PP_ALIGN_LEFT):
    cell = retry_com(lambda: slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, top, width, height))
    style_shape(cell, fill, CARD_BORDER, transparency=0.0)
    set_shape_text(cell, text, size, color=color, bold=bold, align=align, margin_left=8, margin_right=8, margin_top=6, margin_bottom=6)
    return cell


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def generate_assets() -> dict[str, Path]:
    ensure_dirs()
    assets = {
        "bg": ASSET_DIR / "bg_main.png",
        "bg_cover": ASSET_DIR / "bg_cover.png",
        "hook": ASSET_DIR / "hook_visual.png",
        "ai_security": ASSET_DIR / "ai_security_donut.png",
        "core_signals": ASSET_DIR / "core_signals_bar.png",
        "security_spend": ASSET_DIR / "security_spending_line.png",
        "year_one": ASSET_DIR / "year_one_donut.png",
    }
    make_background(assets["bg"], cover=False)
    make_background(assets["bg_cover"], cover=True)
    make_hook_visual(assets["hook"])
    make_donut_chart(
        assets["ai_security"],
        values=[55, 45],
        labels=["Secure", "Failed security tests"],
        colors=[ACCENT, WARNING],
        title="AI-generated Code Security",
    )
    make_bar_chart(
        assets["core_signals"],
        "Core Signals Behind AIE",
        [
            ("Use or plan to use AI tools", 84, ACCENT),
            ("Professional developers use AI daily", 51, ACCENT_SOFT),
            ("Developers distrust AI output", 46, (122, 166, 255)),
            ("AI code failed security tests", 45, WARNING),
            ("Breaches involving the human element", 60, WARNING_SOFT),
        ],
    )
    make_line_chart(assets["security_spend"])
    make_donut_chart(
        assets["year_one"],
        values=[55, 18, 10, 12, 5],
        labels=["Product and research", "Cloud and testing", "Legal and compliance", "Marketing and sales", "Support and pilot"],
        colors=[ACCENT, ACCENT_SOFT, (107, 130, 240), WARNING_SOFT, SUCCESS],
        title="Year-one Cost Allocation",
    )
    return assets


def render_title_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg_cover"])
    pill = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 50, 42, 234, 26)
    style_shape(pill, CARD, ACCENT, transparency=0.0)
    set_shape_text(pill, "BCU ENTERPRISE PRACTICE PROJECT", 10, color=ACCENT, bold=True, align=constants.ppAlignCenter, margin_left=0, margin_right=0, margin_top=3, margin_bottom=0)
    title = add_textbox(slide, 48, 92, 480, 72, spec.title, 30, color=WHITE, bold=True)
    subtitle = add_textbox(slide, 48, 162, 440, 70, spec.content_lines[0], 20, color=MUTED)

    info_card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 48, 276, 348, 164)
    style_shape(info_card, CARD, CARD_BORDER, transparency=0.05)
    info_text = "\r".join(spec.content_lines[1:])
    info = add_textbox(slide, 68, 294, 308, 132, info_text, 17, color=WHITE)
    hero = add_picture(slide, assets["bg_cover"], 514, 78, 380, 380)
    hero.PictureFormat.CropLeft = 1125
    hero.PictureFormat.CropTop = 105
    hero.PictureFormat.CropRight = 240
    hero.PictureFormat.CropBottom = 100

    accent_word = add_textbox(slide, 520, 390, 320, 40, "Autonomous security for the vibe coding era", 16, color=ACCENT, bold=True)

    add_animation(title, constants.ppEffectFade)
    add_animation(subtitle, constants.ppEffectFade)
    add_animation(info_card, constants.ppEffectAppear)
    add_animation(info, constants.ppEffectAppear)
    add_animation(accent_word, constants.ppEffectFade)


def render_hook_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "INTRODUCTION AND HOOK")
    question = add_textbox(slide, 48, 132, 356, 128, spec.content_lines[0], 34, color=WHITE, bold=True)
    supporting = add_textbox(slide, 48, 252, 342, 74, spec.content_lines[1], 18, color=MUTED)
    visual = add_picture(slide, assets["hook"], 410, 120, 496, 295)

    stats = [
        ("84%", "Use or plan to use AI tools"),
        ("51%", "Professional developers use AI daily"),
        ("46%", "Developers distrust AI output"),
    ]
    x = 48
    for idx, (metric, label) in enumerate(stats):
        for shape in add_metric_card(slide, x, 360, 170, 98, metric, label, color=ACCENT if idx < 2 else WARNING_SOFT):
            add_animation(shape, constants.ppEffectAppear)
        x += 186

    source = add_textbox(slide, 48, 472, 300, 18, spec.content_lines[-1], 10, color=MUTED_2)
    add_animation(question, constants.ppEffectFade)
    add_animation(supporting, constants.ppEffectAppear)
    add_animation(visual, constants.ppEffectFade)
    add_animation(source, constants.ppEffectAppear)


def render_problem_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "PROBLEM")
    thesis = add_textbox(slide, 48, 116, 512, 42, spec.content_lines[0], 18, color=MUTED)
    chart = add_picture(slide, assets["ai_security"], 42, 162, 402, 240)

    stat_a = add_metric_card(slide, 48, 412, 186, 86, "60%", "Breaches still involve the human element", color=WARNING_SOFT)
    stat_b = add_metric_card(slide, 254, 412, 186, 86, "34%", "Rise in vulnerability exploitation", color=WARNING)

    chips = [
        ("AUTH", "Insecure authentication"),
        ("SQL", "Injection risk"),
        ("SECRETS", "Exposed secrets"),
        ("CRYPTO", "Weak cryptography"),
        ("CLOUD", "Misconfigured cloud"),
    ]
    positions = [(500, 172), (710, 172), (500, 256), (710, 256), (500, 340)]
    widths = [184, 184, 184, 184, 394]
    chip_shapes = []
    for (badge, label), (left, top), width in zip(chips, positions, widths):
        chip = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, top, width, 66)
        style_shape(chip, CARD, CARD_BORDER, transparency=0.04)
        marker = slide.Shapes.AddShape(constants.msoShapeRectangle, left + 12, top + 12, 8, 42)
        style_shape(marker, WARNING)
        badge_text = add_textbox(slide, left + 28, top + 10, width - 40, 20, badge, 11, color=WARNING_SOFT, bold=True)
        label_text = add_textbox(slide, left + 28, top + 28, width - 40, 26, label, 14, color=WHITE)
        chip_shapes.extend([chip, marker, badge_text, label_text])

    add_animation(thesis, constants.ppEffectAppear)
    add_animation(chart, constants.ppEffectFade)
    for shape in stat_a + stat_b + chip_shapes:
        add_animation(shape, constants.ppEffectAppear)


def render_affected_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "WHO IS AFFECTED")
    cards = [
        ("DEV", "Developers", "Using AI-generated code and fast delivery workflows."),
        ("OPS", "DevOps", "Managing deployment, cloud access, and infrastructure risk."),
        ("SME", "Startups and SMEs", "Building fast without large security teams."),
        ("ENT", "Enterprise teams", "Balancing complexity, compliance, and scale."),
        ("SOC", "Security teams", "Overloaded by alerts, vulnerabilities, and manual checks."),
    ]
    coords = [(48, 134), (334, 134), (620, 134), (190, 288), (476, 288)]
    shapes = []
    for (badge, title, body), (left, top) in zip(cards, coords):
        shapes.extend(add_badge_card(slide, left, top, 244, 120, badge, title, body))
    closing = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 82, 448, 796, 54)
    style_shape(closing, CARD_ALT, ACCENT, transparency=0.02)
    closing_text = add_textbox(
        slide,
        106,
        462,
        748,
        28,
        "The people affected are not just security teams. It is every organisation shipping software faster than it can secure it.",
        16,
        color=WHITE,
        align=constants.ppAlignCenter,
    )
    for shape in shapes:
        add_animation(shape, constants.ppEffectAppear)
    add_animation(closing, constants.ppEffectFade)
    add_animation(closing_text, constants.ppEffectAppear)


def render_why_now_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "WHY IT MATTERS NOW")
    quote = add_textbox(slide, 48, 118, 520, 30, spec.content_lines[0], 16, color=MUTED)
    cards = [
        ("$4.88M", "Average global breach cost in 2024", ACCENT),
        ("$10.5T", "Predicted annual cybercrime cost by 2025", ACCENT_SOFT),
        ("90%", "Breaches linked to misconfigurations or security gaps", WARNING),
        ("4.76M", "Global cybersecurity workforce gap", SUCCESS),
    ]
    positions = [(48, 166), (278, 166), (508, 166), (738, 166)]
    metric_shapes = []
    for (metric, label, color), (left, top) in zip(cards, positions):
        metric_shapes.extend(add_metric_card(slide, left, top, 174, 122, metric, label, color=color))
    strip = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 144, 332, 672, 72)
    style_shape(strip, CARD_ALT, CARD_BORDER, transparency=0.03)
    strip_text = add_textbox(
        slide,
        168,
        354,
        624,
        30,
        "87% of attacks span two or more attack surfaces, blending cloud, identity, SaaS, endpoints, and people.",
        16,
        color=WHITE,
        align=constants.ppAlignCenter,
    )
    source = add_textbox(slide, 50, 472, 620, 18, "Sources: IBM 2024, Cybersecurity Ventures, Palo Alto Networks Unit 42 (Feb 17, 2026), ISC2 2024", 10, color=MUTED_2)
    add_animation(quote, constants.ppEffectAppear)
    for shape in metric_shapes:
        add_animation(shape, constants.ppEffectAppear)
    add_animation(strip, constants.ppEffectFade)
    add_animation(strip_text, constants.ppEffectAppear)
    add_animation(source, constants.ppEffectAppear)


def render_solution_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "SOLUTION")
    summary = add_textbox(slide, 48, 118, 620, 44, spec.content_lines[0], 18, color=MUTED)

    center = slide.Shapes.AddShape(constants.msoShapeOval, 378, 194, 204, 204)
    style_shape(center, CARD_ALT, ACCENT, transparency=0.0)
    set_shape_text(center, "AIE", 28, color=WHITE, bold=True, align=constants.ppAlignCenter, vertical_anchor=constants.msoAnchorMiddle)

    steps = [
        (382, 126, "DETECT"),
        (612, 204, "LEARN"),
        (548, 402, "SIMULATE"),
        (230, 402, "FIX"),
        (152, 204, "IMPROVE"),
    ]
    step_shapes = []
    for left, top, label in steps:
        node = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, top, 124, 44)
        style_shape(node, CARD, ACCENT, transparency=0.02)
        set_shape_text(node, label, 14, color=WHITE, bold=True, align=constants.ppAlignCenter, margin_left=0, margin_right=0, margin_top=9, margin_bottom=0)
        step_shapes.append(node)

    chips = ["Secure code", "Secure cloud", "Learn from attackers", "Validate defences"]
    x = 170
    chip_shapes = []
    for chip in chips:
        box = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, x, 454, 150, 34)
        style_shape(box, CARD_ALT, CARD_BORDER, transparency=0.02)
        font_color = ACCENT if chip != "Validate defences" else WHITE
        set_shape_text(box, chip, 12, color=font_color, bold=True, align=constants.ppAlignCenter, margin_left=0, margin_right=0, margin_top=7, margin_bottom=0)
        chip_shapes.append(box)
        x += 160

    add_animation(summary, constants.ppEffectAppear)
    add_animation(center, constants.ppEffectFade)
    for shape in step_shapes + chip_shapes:
        add_animation(shape, constants.ppEffectWipeRight)


def render_layers_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "PRODUCT")
    quadrants = [
        ("LAYER 1", "Secure Coding Assistant", "Detects injection, auth failures, secrets exposure, and weak cryptography during development."),
        ("LAYER 2", "Cloud Misconfiguration Scanner", "Monitors risky permissions, exposed services, and posture gaps across cloud infrastructure."),
        ("LAYER 3", "Honeypot Intelligence Layer", "Deploys decoys to safely capture and learn from real attacker behaviour."),
        ("LAYER 4", "Red vs Blue Simulation Engine", "Runs continuous validation to prove whether defences actually work."),
    ]
    coords = [(48, 130), (498, 130), (48, 292), (498, 292)]
    shapes = []
    for (label, title, body), (left, top) in zip(quadrants, coords):
        card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, top, 412, 134)
        style_shape(card, CARD, CARD_BORDER, transparency=0.03)
        stripe = slide.Shapes.AddShape(constants.msoShapeRectangle, left, top, 412, 6)
        style_shape(stripe, ACCENT if "1" in label or "2" in label else ACCENT_SOFT)
        label_shape = add_textbox(slide, left + 18, top + 14, 120, 20, label, 12, color=ACCENT, bold=True)
        title_shape = add_textbox(slide, left + 18, top + 34, 360, 26, title, 18, color=WHITE, bold=True)
        body_shape = add_textbox(slide, left + 18, top + 66, 372, 56, body, 15, color=MUTED)
        shapes.extend([card, stripe, label_shape, title_shape, body_shape])
    strip = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 60, 454, 840, 36)
    style_shape(strip, CARD_ALT, ACCENT, transparency=0.02)
    strip_text = add_textbox(slide, 70, 463, 820, 18, "OWASP Top 10 alignment: injection, cryptographic failures, security misconfiguration, and identification/authentication failures.", 11, color=WHITE, align=constants.ppAlignCenter)
    for shape in shapes:
        add_animation(shape, constants.ppEffectFade)
    add_animation(strip, constants.ppEffectAppear)
    add_animation(strip_text, constants.ppEffectAppear)


def render_how_it_works_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "HOW IT WORKS")
    lead = add_textbox(slide, 48, 118, 580, 34, "AIE becomes more useful over time because each layer teaches the others.", 18, color=MUTED)
    flow_labels = [
        "Developer / AI code",
        "Code scanner",
        "Cloud scanner",
        "Honeypot",
        "Simulation",
        "Remediation dashboard",
    ]
    boxes = []
    left = 48
    widths = [140, 126, 126, 116, 116, 186]
    colors = [ACCENT_SOFT, ACCENT, ACCENT, WARNING_SOFT, WARNING, SUCCESS]
    for label, width, color in zip(flow_labels, widths, colors):
        box = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left, 224, width, 92)
        style_shape(box, CARD, CARD_BORDER, transparency=0.03)
        tag = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, left + 12, 236, width - 24, 18)
        style_shape(tag, color)
        text = add_textbox(slide, left + 14, 264, width - 28, 42, label, 16, color=WHITE, bold=True, align=constants.ppAlignCenter)
        boxes.extend([box, tag, text])
        left += width + 12

    arrow_left = 188
    arrows = []
    for _ in range(5):
        arrow = slide.Shapes.AddShape(constants.msoShapeChevron, arrow_left, 258, 36, 26)
        style_shape(arrow, ACCENT)
        arrows.append(arrow)
        arrow_left += 138

    loop = slide.Shapes.AddShape(constants.msoShapeArc, 734, 328, 126, 84)
    loop.Line.ForeColor.RGB = rgb(*ACCENT)
    loop.Line.Weight = 3
    loop.Fill.Visible = False
    add_animation(lead, constants.ppEffectAppear)
    for shape in boxes + arrows + [loop]:
        add_animation(shape, constants.ppEffectWipeRight)

    steps_card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 138, 366, 680, 100)
    style_shape(steps_card, CARD_ALT, CARD_BORDER, transparency=0.02)
    steps_text = add_textbox(
        slide,
        160,
        386,
        640,
        70,
        "Detect: scan code and cloud | Learn: capture exploit patterns | Simulate: prove resilience | Fix: suggest or apply remediation | Improve: push new intelligence back into detection",
        14,
        color=WHITE,
        align=constants.ppAlignCenter,
    )
    add_animation(steps_card, constants.ppEffectFade)
    add_animation(steps_text, constants.ppEffectAppear)


def render_benefits_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "BUSINESS BENEFITS")
    benefits = [
        ("RISK", "Reduce Risk and Cost", "Catch flaws early and reduce breach exposure."),
        ("SPEED", "Operational Speed", "Lower manual workload for security and DevOps teams."),
        ("TRUST", "Customer Trust", "Support compliance readiness and protect reputation."),
        ("RESILIENCE", "Long-Term Resilience", "Turn live attack intelligence into stronger future protection."),
    ]
    coords = [(48, 136), (498, 136), (48, 308), (498, 308)]
    shapes = []
    for (badge, title, body), (left, top) in zip(benefits, coords):
        shapes.extend(add_badge_card(slide, left, top, 412, 142, badge, title, body))
    note = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 198, 466, 564, 34)
    style_shape(note, CARD_ALT, ACCENT, transparency=0.02)
    note_text = add_textbox(slide, 210, 474, 540, 18, "IBM 2024: AI and automation can reduce breach costs when used properly.", 11, color=WHITE, align=constants.ppAlignCenter)
    for shape in shapes:
        add_animation(shape, constants.ppEffectAppear)
    add_animation(note, constants.ppEffectFade)
    add_animation(note_text, constants.ppEffectAppear)


def render_competition_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "COMPETITION")
    quote = add_textbox(slide, 48, 118, 700, 30, spec.content_lines[0], 16, color=MUTED)
    headers = ["Tool", "Main focus", "Limitation", "AIE advantage"]
    rows = [
        ["Snyk", "Code security", "Developer and code centric", "Links code findings to cloud, honeypots, and simulation"],
        ["Thinkst Canary", "Deception and honeypots", "Not a full code/cloud remediation platform", "Uses attacker learning to improve wider defences"],
        ["Prisma Cloud", "Cloud security / CNAPP", "Not built around a full attacker-learning loop", "Connects cloud remediation with live attacker insight"],
        ["Tenable", "Exposure management", "Still depends on teams to act", "Aims to suggest and automate fixes"],
        ["AttackIQ", "Security validation", "Does not unify code, cloud, and deception", "Combines validation with code and cloud intelligence"],
    ]
    col_x = [48, 182, 358, 594]
    col_w = [120, 162, 224, 314]
    y = 154
    header_shapes = []
    for x, w, header in zip(col_x, col_w, headers):
        fill = ACCENT if header == "AIE advantage" else CARD_ALT
        color = BG if header == "AIE advantage" else WHITE
        header_shapes.append(make_table_cell(slide, x, y, w, 34, header, fill, color=color, bold=True, align=constants.ppAlignCenter))
    row_shapes = []
    y = 196
    for row in rows:
        for idx, (x, w, text) in enumerate(zip(col_x, col_w, row)):
            fill = CARD if idx < 3 else (19, 49, 86)
            size = 13 if idx != 0 else 14
            row_shapes.append(make_table_cell(slide, x, y, w, 54, text, fill, size=size, bold=idx == 0))
        y += 60
    source = add_textbox(slide, 48, 500, 720, 16, "Sources: Snyk, Thinkst Canary, Prisma Cloud, Tenable, AttackIQ product pages", 10, color=MUTED_2)
    add_animation(quote, constants.ppEffectAppear)
    for shape in header_shapes + row_shapes:
        add_animation(shape, constants.ppEffectFade)
    add_animation(source, constants.ppEffectAppear)


def render_business_model_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "BUSINESS MODEL")
    strap = add_textbox(slide, 48, 118, 760, 28, "Priced by team size, repositories, cloud assets, and infrastructure footprint.", 17, color=MUTED)
    pricing = [
        ("STARTER", "GBP29/month", "For individual developers and small teams.\r- Code scanning\r- Basic dashboard\r- Limited reports"),
        ("GROWTH", "GBP499/month", "For startups and SMEs.\r- Multi-repository support\r- Cloud posture scans\r- Honeypot integration"),
        ("ENTERPRISE", "GBP2,500+/month", "For larger organisations.\r- Full automation\r- Simulation engine\r- Premium support"),
    ]
    x = 48
    pricing_shapes = []
    for title, price, body in pricing:
        card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, x, 160, 254, 196)
        style_shape(card, CARD, CARD_BORDER, transparency=0.03)
        band = slide.Shapes.AddShape(constants.msoShapeRectangle, x, 160, 254, 8)
        style_shape(band, ACCENT if title != "ENTERPRISE" else SUCCESS)
        pricing_shapes.extend(
            [
                card,
                band,
                add_textbox(slide, x + 18, 178, 214, 20, title, 12, color=ACCENT, bold=True),
                add_textbox(slide, x + 18, 204, 214, 34, price, 24, color=WHITE, bold=True),
                add_textbox(slide, x + 18, 246, 214, 92, body, 15, color=MUTED),
            ]
        )
        x += 278

    chart = add_picture(slide, assets["security_spend"], 522, 360, 370, 150)
    market = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 48, 382, 442, 118)
    style_shape(market, CARD_ALT, CARD_BORDER, transparency=0.03)
    market_title = add_textbox(slide, 66, 398, 200, 20, "Market Timing", 12, color=ACCENT, bold=True)
    market_text = add_textbox(
        slide,
        66,
        422,
        388,
        62,
        "Worldwide information security spending continues to grow. Gartner projects $193B in 2024, $213B in 2025, and $240B in 2026.",
        15,
        color=WHITE,
    )
    add_animation(strap, constants.ppEffectAppear)
    for shape in pricing_shapes:
        add_animation(shape, constants.ppEffectAppear)
    add_animation(market, constants.ppEffectFade)
    add_animation(market_title, constants.ppEffectAppear)
    add_animation(market_text, constants.ppEffectAppear)
    add_animation(chart, constants.ppEffectWipeRight)


def render_investment_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "YEAR-ONE INVESTMENT")
    funding = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 48, 144, 344, 324)
    style_shape(funding, CARD, CARD_BORDER, transparency=0.03)
    funding_head = add_textbox(slide, 70, 166, 180, 20, "Investment need", 12, color=ACCENT, bold=True)
    funding_value = add_textbox(slide, 70, 194, 260, 56, "GBP100k - GBP150k", 28, color=WHITE, bold=True)
    funding_sources = add_textbox(
        slide,
        70,
        270,
        280,
        150,
        "Funding sources\r- Founder contribution\r- BCU innovation support\r- Startup grants\r- Seed investment\r- Pilot partner support",
        16,
        color=MUTED,
    )
    chart = add_picture(slide, assets["year_one"], 414, 132, 492, 300)
    note = add_textbox(slide, 430, 444, 456, 46, "Exact allocation: 55% product and security research, 18% cloud and testing, 10% legal and compliance, 12% marketing and sales, 5% support and pilot testing.", 13, color=MUTED, align=constants.ppAlignCenter)
    for shape in [funding, funding_head, funding_value, funding_sources]:
        add_animation(shape, constants.ppEffectAppear)
    add_animation(chart, constants.ppEffectFade)
    add_animation(note, constants.ppEffectAppear)


def render_team_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "TEAM / FIT / CAPABILITY")
    left = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 48, 136, 390, 334)
    style_shape(left, CARD, CARD_BORDER, transparency=0.03)
    title = add_textbox(slide, 68, 156, 280, 20, "Capability areas", 12, color=ACCENT, bold=True)
    bullets = add_textbox(
        slide,
        68,
        186,
        336,
        248,
        "Cybersecurity learning through my degree and practical modules\rCloud security and shared responsibility knowledge\rSecure coding, OWASP risks, and vulnerability awareness\rSecurity operations, incident response, and risk management\rStrong interest in automation, AI security, and developer tools",
        16,
        color=WHITE,
    )

    roadmap_card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 474, 154, 432, 178)
    style_shape(roadmap_card, CARD, CARD_BORDER, transparency=0.03)
    roadmap_title = add_textbox(slide, 494, 174, 160, 18, "Roadmap", 12, color=ACCENT, bold=True)
    roadmap_steps = ["Learning", "MVP", "Pilot", "Adaptive platform"]
    x = 506
    roadmap_shapes = []
    for idx, step in enumerate(roadmap_steps):
        node = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, x, 234, 84 if idx < 3 else 144, 42)
        style_shape(node, CARD_ALT, ACCENT if idx < 3 else SUCCESS, transparency=0.01)
        set_shape_text(node, step, 13, color=WHITE, bold=True, align=constants.ppAlignCenter, margin_left=0, margin_right=0, margin_top=10, margin_bottom=0)
        roadmap_shapes.append(node)
        if idx < 3:
            chevron = slide.Shapes.AddShape(constants.msoShapeChevron, x + node.Width + 8, 242, 24, 24)
            style_shape(chevron, ACCENT)
            roadmap_shapes.append(chevron)
        x += 108 if idx < 2 else 168

    bottom = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 474, 356, 432, 114)
    style_shape(bottom, CARD_ALT, CARD_BORDER, transparency=0.03)
    bottom_text = add_textbox(slide, 494, 382, 396, 62, "This project turns academic cybersecurity learning into a practical, industry-facing solution with a clear pathway from concept to pilot.", 16, color=WHITE)
    for shape in [left, title, bullets, roadmap_card, roadmap_title, bottom, bottom_text] + roadmap_shapes:
        add_animation(shape, constants.ppEffectAppear)


def render_traction_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "TRACTION / FEEDBACK")
    strip = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 48, 120, 858, 64)
    style_shape(strip, CARD_ALT, CARD_BORDER, transparency=0.03)
    strip_text = add_textbox(
        slide,
        72,
        140,
        820,
        28,
        "Feedback improved the concept from a simple security scanner into a continuous adaptive security platform.",
        16,
        color=WHITE,
        align=constants.ppAlignCenter,
    )

    before = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 72, 220, 300, 236)
    style_shape(before, CARD, CARD_BORDER, transparency=0.03)
    before_title = add_textbox(slide, 92, 240, 180, 20, "Before", 14, color=WARNING_SOFT, bold=True)
    before_text = add_textbox(slide, 92, 272, 250, 128, "Simple scanner\r- One-off detection\r- Limited differentiation\r- Lower strategic value", 18, color=WHITE)

    arrow = slide.Shapes.AddShape(constants.msoShapeChevron, 416, 294, 116, 82)
    style_shape(arrow, ACCENT)
    arrow_text = add_textbox(slide, 432, 318, 80, 30, "FEEDBACK", 14, color=BG, bold=True, align=constants.ppAlignCenter)

    after = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 576, 220, 300, 236)
    style_shape(after, CARD, CARD_BORDER, transparency=0.03)
    after_title = add_textbox(slide, 596, 240, 180, 20, "After", 14, color=SUCCESS, bold=True)
    after_text = add_textbox(
        slide,
        596,
        272,
        250,
        136,
        "Autonomous adaptive platform\r- Continuous learning\r- Cloud remediation\r- Honeypot intelligence\r- Red-vs-blue validation",
        18,
        color=WHITE,
    )
    for shape in [strip, strip_text, before, before_title, before_text]:
        add_animation(shape, constants.ppEffectAppear)
    for shape in [arrow, arrow_text, after, after_title, after_text]:
        add_animation(shape, constants.ppEffectWipeRight)


def render_ask_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg_cover"])
    add_title_block(slide, spec.title, "THE ASK")
    ask = add_textbox(slide, 48, 124, 726, 46, "Support a 12-month pilot of AIE with funding, mentorship, and access to a safe cloud test environment.", 24, color=WHITE, bold=True)

    needs, needs_head, needs_body = add_card(
        slide,
        48,
        204,
        388,
        188,
        "WHAT I NEED",
        "Pilot funding\rCybersecurity mentor feedback\rAccess to a safe cloud test environment\rPilot users from startups, SMEs, or innovation partners",
        accent_color=ACCENT,
        body_size=16,
    )
    success, success_head, success_body = add_card(
        slide,
        466,
        204,
        388,
        188,
        "WHAT SUCCESS LOOKS LIKE",
        "Working MVP\rTested code and cloud scanning workflow\rHoneypot intelligence prototype\rClear pilot feedback\rEvidence for future investment",
        accent_color=SUCCESS,
        body_size=16,
    )
    closing = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 48, 422, 806, 48)
    style_shape(closing, CARD_ALT, ACCENT, transparency=0.02)
    closing_text = add_textbox(slide, 72, 436, 760, 22, "Security can't stay manual. It has to be autonomous.", 19, color=WHITE, bold=True, align=constants.ppAlignCenter)
    contact = add_textbox(slide, 48, 486, 420, 18, "Ibrahim Timilehin  |  ibrahim.timilehin@bcu.ac.uk", 11, color=WHITE)
    hero = add_picture(slide, assets["bg_cover"], 760, 60, 180, 180)
    hero.PictureFormat.CropLeft = 1330
    hero.PictureFormat.CropTop = 200
    hero.PictureFormat.CropRight = 340
    hero.PictureFormat.CropBottom = 480
    for shape in [ask, needs, needs_head, needs_body, success, success_head, success_body, closing, closing_text, contact]:
        add_animation(shape, constants.ppEffectAppear)


def render_sources_slide(slide, spec: SlideSpec, assets: dict[str, Path]):
    add_background(slide, assets["bg"])
    add_title_block(slide, spec.title, "DATA SOURCES / REFERENCES")
    intro = add_textbox(slide, 48, 118, 760, 26, spec.content_lines[0], 15, color=MUTED)

    left_card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 48, 152, 404, 326)
    right_card = slide.Shapes.AddShape(constants.msoShapeRoundedRectangle, 500, 152, 404, 326)
    style_shape(left_card, CARD, CARD_BORDER, transparency=0.03)
    style_shape(right_card, CARD, CARD_BORDER, transparency=0.03)

    left_sources = [
        "Stack Overflow Developer Survey 2025\nsurvey.stackoverflow.co/2025/ai",
        "Veracode GenAI Code Security Report 2025\nveracode.com/blog/genai-code-security-report",
        "IBM Cost of a Data Breach 2024\nibm.com/think/insights/whats-new-2024-cost-of-a-data-breach-report",
        "Verizon 2025 DBIR\nverizon.com/about/news/2025-data-breach-investigations-report",
        "Verizon DBIR human element reference\nverizon.com/business/resources/articles/s/how-to-protect-your-organization-from-a-pretexting-attack",
        "Gartner spending forecast 2025\ngartner.com/en/newsroom/press-releases/2025-07-29-gartner-forecasts-worldwide-end-user-spending-on-information-security-to-total-213-billion-us-dollars-in-2025",
    ]
    right_sources = [
        "ISC2 Cybersecurity Workforce Study 2024\nisc2.org/insights/2024/10/isc2-2024-cybersecurity-workforce-study",
        "Palo Alto Networks Unit 42 press release (Feb 17, 2026)\ninvestors.paloaltonetworks.com/news-releases/news-release-details/unit-42-report-ai-and-attack-surface-complexity-fuel-majority",
        "OWASP Top 10\nowasp.org/Top10",
        "Snyk product page\nsnyk.io/product",
        "Thinkst Canary\ncanary.tools",
        "Prisma Cloud / Tenable / AttackIQ\npaloaltonetworks.com/prisma/cloud | tenable.com/products | attackiq.com",
    ]
    left_text = add_textbox(slide, 68, 174, 364, 286, "\r\r".join(left_sources), 11, color=WHITE)
    right_text = add_textbox(slide, 520, 174, 364, 286, "\r\r".join(right_sources), 11, color=WHITE)
    bottom = add_textbox(slide, 48, 488, 760, 18, "These sources show why AIE is timely, relevant, and connected to current industry needs.", 11, color=MUTED, align=constants.ppAlignCenter)
    for shape in [intro, left_card, right_card, left_text, right_text, bottom]:
        add_animation(shape, constants.ppEffectFade)


def render_slide(presentation, spec: SlideSpec, assets: dict[str, Path]):
    slide = presentation.Slides.Add(spec.number, PP_LAYOUT_BLANK)
    set_slide_transition(slide)
    renderers = {
        "title": render_title_slide,
        "hook": render_hook_slide,
        "problem": render_problem_slide,
        "affected": render_affected_slide,
        "why_now": render_why_now_slide,
        "solution": render_solution_slide,
        "layers": render_layers_slide,
        "how_it_works": render_how_it_works_slide,
        "benefits": render_benefits_slide,
        "competition": render_competition_slide,
        "business_model": render_business_model_slide,
        "investment": render_investment_slide,
        "team": render_team_slide,
        "traction": render_traction_slide,
        "ask": render_ask_slide,
        "sources": render_sources_slide,
    }
    renderers[spec.layout_key](slide, spec, assets)
    add_footer(slide, spec.number)
    add_notes(slide, spec)


def write_guide():
    sections = [
        "# AIE Enterprise Practice Pitch Deck",
        "",
        "Professional pitch deck outline for the BCU Enterprise Practice Project.",
        "",
        "## Project",
        "",
        "- Title: AIE - Autonomous Security Engineer in the Cloud",
        "- Presenter: Ibrahim Timilehin",
        "- Core message: Security can't stay manual. It has to be autonomous.",
        "",
    ]
    for spec in SLIDES:
        sections.extend(
            [
                f"## Slide {spec.number} - {spec.title}",
                "",
                f"Slide title: {spec.title}",
                "",
                "Slide content:",
                *[f"- {line}" for line in spec.content_lines],
                "",
                f"Suggested visual or chart: {spec.visual}",
                "",
                f"Speaker notes: {spec.speaker_notes}",
                "",
                f"Suggested animation: {spec.animation}",
                "",
            ]
        )

    script = " ".join(spec.script_segment for spec in SLIDES[:-1])
    sections.extend(
        [
            "## Three-minute Speaker Script",
            "",
            script,
            "",
            "## Verified Sources",
            "",
        ]
    )
    for name, url, note in SOURCES:
        sections.extend([f"- {name}: {url}", f"  {note}"])
    GUIDE_PATH.write_text("\n".join(sections), encoding="utf-8")


def export_previews(presentation):
    for file in PREVIEW_DIR.glob("*"):
        if file.is_file():
            file.unlink()
    presentation.Export(str(PREVIEW_DIR.resolve()), "PNG", 1600, 900)


def build_deck():
    ensure_dirs()
    assets = generate_assets()
    write_guide()

    if PPT_PATH.exists():
        PPT_PATH.unlink()

    powerpoint = win32.gencache.EnsureDispatch("PowerPoint.Application")
    powerpoint.Visible = True
    time.sleep(1.0)
    presentation = powerpoint.Presentations.Add()
    time.sleep(0.6)
    for _ in range(4):
        try:
            presentation.PageSetup.SlideSize = PP_SLIDE_SIZE_16X9
            break
        except Exception:
            time.sleep(0.5)
    presentation.BuiltInDocumentProperties("Title").Value = "AIE Enterprise Practice Pitch Deck"
    presentation.BuiltInDocumentProperties("Author").Value = "Ibrahim Timilehin"
    presentation.BuiltInDocumentProperties("Subject").Value = "BCU Enterprise Practice Project"
    presentation.BuiltInDocumentProperties("Comments").Value = "Generated by tools/generate_aie_pitch_deck.py"

    try:
        for spec in SLIDES:
            render_slide(presentation, spec, assets)
        presentation.SaveAs(str(PPT_PATH.resolve()))
        export_previews(presentation)
    finally:
        presentation.Close()
        powerpoint.Quit()


if __name__ == "__main__":
    build_deck()
