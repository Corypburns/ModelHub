import logging
import random
import time

from playwright.sync_api import sync_playwright, Page

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
# This format ensures you always know exactly which function and line number 
# generated the log, along with the timestamp and severity level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Target URLs
# ──────────────────────────────────────────────────────────────────────────────

URLS = [
    "https://example.com",
    "https://news.ycombinator.com",
    "https://www.wikipedia.org",
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://www.reuters.com",
    "https://github.com",
    "https://stackoverflow.com",
    "https://www.nytimes.com",
    "https://www.mozilla.org",
]

# ──────────────────────────────────────────────────────────────────────────────
# Interaction helpers
# ──────────────────────────────────────────────────────────────────────────────

def smooth_scroll(page: Page, direction: str = "down", steps: int = 2, delay: float = 0.12):
    """Scroll incrementally to simulate a real user."""
    logger.debug(f"Starting smooth scroll. Direction: {direction}, Steps: {steps}")
    try:
        if direction == "down":
            total = page.evaluate("document.body.scrollHeight - window.innerHeight")
            current = page.evaluate("window.scrollY")
            delta = (total - current) / steps
        else:
            current = page.evaluate("window.scrollY")
            delta = -current / steps

        for i in range(steps):
            page.evaluate(f"window.scrollBy(0, {int(delta)})")
            time.sleep(delay)
    except Exception as e:
        logger.error(f"Failed to execute smooth scroll ({direction}): {e}", exc_info=True)


def hover_links(page: Page, count: int = 8):
    """Hover over visible links to simulate mouse movement."""
    logger.debug(f"Attempting to hover over up to {count} links")
    try:
        links = page.query_selector_all("a[href]")[:count * 2]
        hovered = 0
        for link in links:
            if hovered >= count:
                break
            try:
                if link.is_visible():
                    link.hover(timeout=500)
                    time.sleep(0.1)
                    hovered += 1
            except Exception as e:
                # Using warning here as missing a single hover isn't a critical failure
                logger.warning(f"Error hovering over a specific link: {e}", exc_info=True)
                continue
        logger.info(f"Successfully hovered over {hovered} links.")
    except Exception as e:
        logger.error(f"Failed to query links for hovering: {e}", exc_info=True)


def click_and_back(page: Page) -> bool:
    """
    Click a safe visible link then navigate back.
    Returns True if a click was successfully performed.
    """
    selectors = ["nav a", "a[href]", "[role='link']"]
    for selector in selectors:
        logger.debug(f"Trying to find clickable link with selector: '{selector}'")
        try:
            elements = page.query_selector_all(selector)
            visible = [
                el for el in elements
                if el.is_visible() and el.bounding_box() is not None
            ]
            if not visible:
                logger.debug(f"No visible elements found for selector '{selector}'")
                continue
            
            target = random.choice(visible[:10])
            target.click(timeout=500)
            logger.info(f"Clicked element using selector '{selector}'. Navigating back...")
            time.sleep(1.0)
            
            page.go_back(timeout=2000)
            page.wait_for_load_state("load", timeout=2000)
            return True
        except Exception as e:
            logger.warning(f"Failed interacting with selector '{selector}': {e}", exc_info=True)
            continue
            
    logger.info("Could not find a suitable link to click across any configured selector.")
    return False


def type_in_search(page: Page) -> bool:
    """
    Find a search input, type a query slowly, then clear it.
    Returns True if a search box was found and interacted with.
    """
    selectors = [
        "input[type='search']",
        "input[type='text']",
        "[role='searchbox']",
        "input[placeholder*='search' i]",
        "input[placeholder*='Search' i]",
    ]
    for sel in selectors:
        logger.debug(f"Looking for search box using selector: '{sel}'")
        try:
            box = page.query_selector(sel)
            if box and box.is_visible():
                box.click(timeout=1000)
                box.type("open source software", delay=70)   # 70 ms per char
                logger.info(f"Successfully typed in search box using selector '{sel}'.")
                time.sleep(0.5)
                box.triple_click()
                box.fill("")
                return True
        except Exception as e:
            logger.warning(f"Failed to interact with search box using selector '{sel}': {e}", exc_info=True)
            continue
            
    logger.info("No interactive search box found on the page.")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Per-URL interaction sequence
# ──────────────────────────────────────────────────────────────────────────────

def interact(page: Page, url: str):
    """Full interaction sequence for one URL."""
    # 1. Load page
    logger.info(f"Loading URL: {url}")
    page.goto(url, wait_until="load", timeout=10000) # Increased timeout safety to 30s as 3s is very low
    time.sleep(0.5)

    # 2. Scroll to bottom
    logger.info("Scrolling down...")
    smooth_scroll(page, direction="down", steps=10, delay=0.15)
    time.sleep(0.3)

    # 3. Pause mid-page (simulate reading)
    logger.info("Pausing mid-page to simulate reading (2s)...")
    page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
    time.sleep(2.0)

    # 4. Hover over links
    logger.info("Initiating link hover behavior...")
    hover_links(page, count=8)
    time.sleep(0.2)

    # 5. Click a link and navigate back
    logger.info("Attempting to click a link and navigate back...")
    success = click_and_back(page)
    logger.info(f"Click & back result: {'SUCCESS' if success else 'FAILED (No suitable link)'}")
    time.sleep(0.3)

    # 6. Scroll back to top
    logger.info("Scrolling up...")
    smooth_scroll(page, direction="up", steps=10, delay=0.15)
    time.sleep(0.3)

    # 7. Search box interaction
    logger.info("Attempting to find and interact with search input...")
    found = type_in_search(page)
    logger.info(f"Search box interaction result: {'SUCCESS' if found else 'FAILED (None found)'}")
    time.sleep(0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Main workload runner
# ──────────────────────────────────────────────────────────────────────────────

def run(size: int = 1, urls: list[str] = URLS, delay: float = 0.0, inference_timer=None):
    """
    Run the full interaction sequence over `urls`, `size` times.

    Parameters
    ----------
    urls : list of URL strings to visit each cycle
    size : number of full cycles (higher = longer run, better for energy averaging)
    delay : delay between URL visits
    inference_timer : InferenceTimer instance for timing
    """
    logger.info(f"Starting automation run. Total URLs: {len(urls)}, Cycles: {size}")

    with sync_playwright() as p:
        logger.info("Launching chromium browser (headless)...")
        browser = p.chromium.launch(headless=True)

        print(f'Type of size is {type(size)} and size is {size}')
        for cycle in range(int(size/4)):
            logger.info(f"{'='*60}")
            logger.info(f"Cycle {cycle + 1} / {size}")
            logger.info(f"{'='*60}")

            inference_timer.start_cycle() if inference_timer else None

            for url in urls:
                logger.info(f"── Starting interaction loop for: {url}")
                page = browser.new_page(
                    viewport={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                )
                try:
                    interact(page, url)
                    logger.info(f"Successfully completed interaction loop for: {url}")
                except Exception as e:
                    logger.error(f"Critical error during interaction loop for {url}: {e}", exc_info=True)
                finally:
                    logger.debug(f"Closing page for URL: {url}")
                    page.close()

            inference_timer.end_cycle() if inference_timer else None

        logger.info("Closing browser...")
        browser.close()

    logger.info("Automation run completed.")

if __name__ == "__main__":
    # Execute the script
    run()