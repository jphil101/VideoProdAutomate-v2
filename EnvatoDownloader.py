"""
Envato Elements Browser Automation Downloader

Automates searching and downloading full-quality, licensed videos
from Envato Elements (app.envato.com) using CloakBrowser.

First run: Will open a visible browser and wait for you to log in.
Subsequent runs: Will use the saved session to run automatically.
"""

import os
import re
import sys
import time
import random
import argparse
from pathlib import Path

def get_cross_platform_cache_dir(app_name):
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, app_name)
    elif sys.platform == "darwin":
        return os.path.expanduser(f"~/Library/Caches/{app_name}")
    else:
        return os.path.expanduser(f"~/.cache/{app_name}")

def bypass_playwright_node_execution_restriction():
    try:
        # Resolve playwright's location dynamically
        import importlib.util
        spec = importlib.util.find_spec("playwright")
        if spec and spec.origin:
            driver_dir = os.path.join(os.path.dirname(spec.origin), "driver")
            node_exe = "node.exe" if sys.platform == "win32" else "node"
            src_node = os.path.join(driver_dir, node_exe)
            
            if os.path.exists(src_node):
                cache_dir = get_cross_platform_cache_dir("playwright_node_cache")
                os.makedirs(cache_dir, exist_ok=True)
                dest_node = os.path.join(cache_dir, node_exe)
                
                if not os.path.exists(dest_node) or os.path.getsize(src_node) != os.path.getsize(dest_node):
                    import shutil
                    shutil.copy2(src_node, dest_node)
                    
                if sys.platform != "win32":
                    import stat
                    st = os.stat(dest_node)
                    os.chmod(dest_node, st.st_mode | stat.S_IEXEC)
                    
                os.environ["PLAYWRIGHT_NODEJS_PATH"] = dest_node
    except Exception as e:
        pass

bypass_playwright_node_execution_restriction()

try:
    from cloakbrowser import launch_persistent_context
    from playwright.sync_api import TimeoutError
except ImportError:
    print("ERROR: Missing dependencies. Please run:")
    print("       pip install cloakbrowser playwright")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_N = 5
DOWNLOAD_TIMEOUT_MS = 300000  # 5 minutes

SCRIPT_DIR = Path(__file__).resolve().parent
PENDRIVE_PROFILE_DIR = SCRIPT_DIR / ".envato_session"
LOCAL_PROFILE_DIR = Path(get_cross_platform_cache_dir("envato_sessions")) / "active_session"
BASE_DOWNLOAD_DIR = SCRIPT_DIR / "downloads"

# Ensure base download directory exists
BASE_DOWNLOAD_DIR.mkdir(exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe for filenames."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80]


def human_delay(min_sec=1.0, max_sec=3.0):
    """Add a random delay to seem more human."""
    time.sleep(random.uniform(min_sec, max_sec))


class EnvatoElementsDownloader:
    def __init__(self, download_dir: Path, headless=False, media_type="video"):
        self.headless = headless
        self.context = None
        self.page = None
        self.download_dir = download_dir
        self.media_type = media_type

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        """Launch the stealth browser with persistent session."""
        print(f"🚀 Preparing local cache for browser profile...")
        
        # Clean any stale lock files from previous runs to prevent SingletonLock/existing session errors
        for lock_dir in [PENDRIVE_PROFILE_DIR, LOCAL_PROFILE_DIR]:
            if lock_dir.exists():
                for root, dirs, files in os.walk(lock_dir):
                    for f in files:
                        if "lock" in f.lower() or f.startswith("Singleton"):
                            try:
                                os.remove(os.path.join(root, f))
                            except:
                                pass
                                
        # Copy profile from FAT32 pendrive to native filesystem to avoid SingletonLock errors
        if PENDRIVE_PROFILE_DIR.exists():
            import shutil
            shutil.copytree(PENDRIVE_PROFILE_DIR, LOCAL_PROFILE_DIR, dirs_exist_ok=True)
            
        print(f"🚀 Launching CloakBrowser...")
        
        # We need a persistent context so we don't have to login every time.
        # humanize=True adds realistic mouse movements and typing delays.
        # We handle downloads explicitly by accepting downloads.
        try:
            self.context = launch_persistent_context(
                LOCAL_PROFILE_DIR,
                headless=self.headless,
                humanize=True,
                accept_downloads=True,
                args=["--window-size=1920,1080"],
                viewport={"width": 1920, "height": 1080}
            )
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                print("\n   [!] Missing Playwright browser binaries on this computer.")
                print("   [!] Auto-installing them to your OS native cache folder now (this may take a minute)...")
                import subprocess
                subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                print("   ✅ Playwright browsers installed successfully! Retrying launch...\n")
                self.context = launch_persistent_context(
                    LOCAL_PROFILE_DIR,
                    headless=self.headless,
                    humanize=True,
                    accept_downloads=True,
                    args=["--window-size=1920,1080"],
                    viewport={"width": 1920, "height": 1080}
                )
            else:
                raise
        self.page = self.context.new_page()

    def stop(self):
        """Close the browser and sync profile back to pendrive."""
        if self.context:
            self.context.close()
            print("💾 Saving browser session back to pendrive...")
            import shutil
            # We ignore specific lock files and large cache directories to significantly speed up saving to FAT32 pendrives
            shutil.copytree(
                LOCAL_PROFILE_DIR, 
                PENDRIVE_PROFILE_DIR, 
                dirs_exist_ok=True, 
                ignore=shutil.ignore_patterns(
                    "Singleton*", 
                    "Cache", 
                    "CacheStorage", 
                    "Code Cache", 
                    "GPUCache", 
                    "DawnCache", 
                    "Crashpad", 
                    "Crash Reports", 
                    "Network Action Predictor"
                )
            )
    def handle_cookie_consent(self):
        """Find and click 'Accept all' cookie consent button if present to prevent page overlay issues."""
        try:
            accept_btn = self.page.locator("button:visible", has_text=re.compile(r"Accept\s*all", re.IGNORECASE)).first
            if accept_btn.is_visible():
                print("   🍪 Cookie consent banner detected. Clicking Accept all...")
                accept_btn.click()
                human_delay(1, 2)
        except Exception:
            pass

    def ensure_logged_in(self):
        """Check if logged in, otherwise pause for manual login."""
        self.page.goto("https://app.envato.com/", wait_until="domcontentloaded")
        self.handle_cookie_consent()
        # Give the page time to fully render dynamic elements (avatar, sign-in button)
        human_delay(3, 5)
        
        print("🔍 Checking Envato login status...")
        try:
            # Look for the user profile button/avatar in the nav bar
            # Alternatively, check that "Sign In" is absent
            profile_btn = self.page.locator("button[data-test-selector='user-menu-trigger']").first
            sign_in_btn = self.page.locator("a[href*='sign_in']").first
            
            # Wait up to 5 seconds to see which one appears
            try:
                self.page.wait_for_selector("button[data-test-selector='user-menu-trigger'], a[href*='sign_in']", timeout=5000)
            except TimeoutError:
                pass
                
            if profile_btn.is_visible() and not sign_in_btn.is_visible():
                print("✅ Auto-detected active Envato session! Proceeding...")
                return
        except Exception as e:
            print(f"⚠️ Auto-detection encountered an error: {e}")

        print("\n" + "!" * 60)
        print("  ACTION REQUIRED:")
        print("  Envato login could not be automatically verified.")
        print("  Please look at the automated browser window.")
        print("  If you are NOT logged in, please log into Envato Elements now.")
        print("  Once you are fully logged in (or if you already are),")
        print("  PRESS ENTER HERE IN THE TERMINAL TO CONTINUE.")
        print("!" * 60 + "\n")
        
        input("Press Enter to continue searching and downloading... ")
        print("✅ Proceeding with saved session...")
        human_delay(1, 2)

    def search_and_get_links(self, phrase: str, count: int) -> list[str]:
        """Search for items and return URLs."""
        print(f"\n🔍 Searching for {self.media_type}s: '{phrase}'")
        encoded_phrase = phrase.replace(" ", "+")
        if self.media_type == "photo":
            search_url = f"https://app.envato.com/search?itemType=photos&term={encoded_phrase}"
        else:
            search_url = f"https://app.envato.com/search?itemType=stock-video&term={encoded_phrase}&filter.orientation=Vertical"
        self.page.goto(search_url, wait_until="domcontentloaded")
        self.handle_cookie_consent()
        human_delay(3, 5)

        # Scroll down a bit to load results
        self.page.mouse.wheel(0, 500)
        human_delay(1, 2)
        self.page.mouse.wheel(0, 500)
        human_delay(1, 2)

        links_data = []
        try:
            if self.media_type == "photo":
                valid_substrings = ["photo", "image", "picture"]
            else:
                valid_substrings = ["video", "motion-graphic"]
                
            selector = ", ".join([f"a[href*='{sub}']" for sub in valid_substrings])
            self.page.wait_for_selector(selector, timeout=10000)
            
            elements = self.page.locator(selector).all()
            for el in elements:
                href = el.get_attribute("href")
                if href:
                    # Filter for item page URLs containing the 7-12 character uppercase ID pattern or UUIDv4 pattern
                    if re.search(r"-[A-Z0-9]{7,12}(?:[/?]|$)|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", href):
                        if not any(d['raw_url'] == href for d in links_data):
                            title = el.get_attribute("aria-label")
                            if not title:
                                try:
                                    title = el.locator("img").first.get_attribute("alt")
                                except:
                                    pass
                            if not title:
                                try:
                                    title = el.inner_text().strip().split('\n')[0]
                                except:
                                    pass
                            if not title:
                                title = "Unknown Title"
                                
                            links_data.append({"raw_url": href, "title": title.strip()})
                            if len(links_data) >= count:
                                break
        except TimeoutError:
            print(f"⚠️  No {self.media_type} results found or page structure changed.")

        # Ensure absolute URLs and bypass the modal overlay
        absolute_links = []
        for item in links_data:
            link = item["raw_url"]
            # If the link opens the modal overlay (e.g. /search/all/stock-video/...)
            # we strip out the search part to get the standalone item page URL
            clean_link = link.replace("/search/all/", "/")
            clean_link = clean_link.split("?")[0] # remove search tracking parameters
            
            if clean_link.startswith("http"):
                abs_url = clean_link
            else:
                abs_url = f"https://app.envato.com{clean_link}"
                
            absolute_links.append({
                "url": abs_url,
                "title": item["title"]
            })
                
        print(f"✅ Found {len(absolute_links)} {self.media_type} links.")
        return absolute_links

    def download_item(self, index: int, item_url: str):
        """Navigate to an item page, trigger download, and save the file."""
        print(f"\n── Result {index}: {item_url}")
        self.page.goto(item_url, wait_until="domcontentloaded")
        self.handle_cookie_consent()
        human_delay(2, 4)

        # Get item title for filename
        title = f"{self.media_type}_item"
        try:
            title_el = self.page.locator("h1").first
            if title_el.is_visible():
                title = title_el.inner_text()
        except:
            pass

        # We define a base filename without extension first
        base_filename = f"{index}_{sanitize_filename(title)}"
        
        # Click the download button (using a more robust selector that matches 'Download' case-insensitively)
        download_btn = self.page.locator("button:visible", has_text=re.compile(r"\bDownload\b", re.IGNORECASE)).first
        try:
            download_btn.wait_for(state="visible", timeout=30000)
        except TimeoutError:
            print("   ❌ 'Download' button not found after waiting.")
            return "UNAUTHORIZED"

        # The click must be wrapped INSIDE expect_download, otherwise Playwright misses the event 
        # and the browser downloads it natively into the default Downloads folder.
        try:
            print("   ⏳ Triggering download... (waiting up to 60s for server)")
            with self.page.expect_download(timeout=60000) as download_info:
                print("   🖱️  Clicking Download button...")
                download_btn.click(force=True)
                
                project_dialog = self.page.locator("button:visible", has_text=re.compile(r"(Add.*Download|Download without license)", re.IGNORECASE)).first
                try:
                    if project_dialog.wait_for(state="visible", timeout=6000):
                        print("   📝 License dialog detected. Confirming download...")
                        project_dialog.click()
                except TimeoutError:
                    pass
                    
                # Force pause any playing videos to save bandwidth/CPU
                try:
                    self.page.evaluate("document.querySelectorAll('video').forEach(v => v.pause());")
                except:
                    pass
                    
            download = download_info.value
            
            # Use the actual file extension from Envato's server
            original_filename = download.suggested_filename
            ext = os.path.splitext(original_filename)[1]
            if not ext:
                ext = ".mp4" if self.media_type == "video" else ".jpg"  # fallback if no extension provided
                
            final_filename = f"{base_filename}{ext}"
            filepath = self.download_dir / final_filename
            
            print(f"   ⬇️  Downloading file... (saving to {final_filename})")
            
            # Save the download to our folder instead of the default native downloads folder
            download.save_as(filepath)
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ Saved {size_mb:.1f} MB to {filepath}")
            return True
            
        except TimeoutError:
            print("   ❌ Download did not start within the timeout.")
            return False
        except Exception as e:
            print(f"   ❌ Error during download: {e}")
            return False

    def trigger_download(self, index: int, item_url: str):
        """Navigate to an item page and trigger download, returning the Download object without waiting for it to finish."""
        print(f"\n── Result {index}: {item_url}")
        self.page.goto(item_url, wait_until="domcontentloaded")
        self.handle_cookie_consent()
        human_delay(2, 4)

        title = f"{self.media_type}_item"
        try:
            title_el = self.page.locator("h1").first
            if title_el.is_visible():
                title = title_el.inner_text()
        except:
            pass

        base_filename = f"{index}_{sanitize_filename(title)}"
        
        download_btn = self.page.locator("button:visible", has_text=re.compile(r"\bDownload\b", re.IGNORECASE)).first
        try:
            download_btn.wait_for(state="visible", timeout=30000)
        except TimeoutError:
            print("   ❌ 'Download' button not found after waiting.")
            return "UNAUTHORIZED"

        try:
            print("   ⏳ Triggering download... (streaming in background)")
            with self.page.expect_download(timeout=60000) as download_info:
                print("   🖱️  Clicking Download button...")
                download_btn.click(force=True)
                
                project_dialog = self.page.locator("button:visible", has_text=re.compile(r"(Add.*Download|Download without license)", re.IGNORECASE)).first
                try:
                    if project_dialog.wait_for(state="visible", timeout=6000):
                        print("   📝 License dialog detected. Confirming download...")
                        project_dialog.click()
                except TimeoutError:
                    pass
                    
                try:
                    self.page.evaluate("document.querySelectorAll('video').forEach(v => v.pause());")
                except:
                    pass
                    
            download = download_info.value
            
            original_filename = download.suggested_filename
            ext = os.path.splitext(original_filename)[1]
            if not ext:
                ext = ".mp4" if self.media_type == "video" else ".jpg"
                
            final_filename = f"{base_filename}{ext}"
            filepath = self.download_dir / final_filename
            
            return {"download_obj": download, "filepath": filepath, "final_filename": final_filename}
            
        except TimeoutError:
            print("   ❌ Download did not start within the timeout.")
            return False
        except Exception as e:
            print(f"   ❌ Error during download trigger: {e}")
            return False

    def wait_and_save_download(self, download_obj, filepath):
        """Blocks to save a previously triggered download to disk."""
        print(f"   ⬇️  Saving background download to {filepath.name}...")
        try:
            download_obj.save_as(filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ Saved {size_mb:.1f} MB to {filepath}")
            return True
        except Exception as e:
            print(f"   ❌ Error saving download: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Envato Elements CloakBrowser Downloader")
    parser.add_argument("--term", type=str, required=True, help="The exact search phrase (e.g., 'modular building construction')")
    parser.add_argument("--segment", type=str, required=True, help="Unique segment number/id for the subfolder")
    parser.add_argument("--count", type=int, default=TOP_N, help="Number of items to download (default 5)")
    parser.add_argument("--type", type=str, choices=["video", "photo"], default="video", help="Media type to search for")
    args = parser.parse_args()

    search_phrase = args.term
    segment_number = args.segment
    download_count = args.count
    media_type = args.type

    print("=" * 60)
    print("  Envato Elements CloakBrowser Downloader")
    print(f"  Search Term: '{search_phrase}'")
    print(f"  Segment ID:  {segment_number}")
    print("=" * 60)

    # Create segment-specific download directory
    segment_dir = BASE_DOWNLOAD_DIR / segment_number
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw search term to a text file inside the segment directory
    term_file = segment_dir / "search_term.txt"
    term_file.write_text(search_phrase)
    
    # We use headless=False by default so you can see it working
    downloader = EnvatoElementsDownloader(download_dir=segment_dir, headless=False, media_type=media_type)
    
    try:
        downloader.start()
        downloader.ensure_logged_in()
        
        links = downloader.search_and_get_links(search_phrase, download_count)
        
        if not links:
            print("No items to download.")
            return

        success_count = 0
        failed_links = []
        
        # Initial pass
        for i, link in enumerate(links, 1):
            if downloader.download_item(i, link):
                success_count += 1
            else:
                failed_links.append((i, link))
            human_delay(3, 6) # Delay between items
            
        # Auto-retry logic for failed downloads (up to 3 retries)
        max_retries = 3
        current_retry = 0
        
        while failed_links and current_retry < max_retries:
            current_retry += 1
            print(f"\n🔄 Auto-retry pass {current_retry}/{max_retries} for {len(failed_links)} failed downloads...")
            
            # Copy current failures to process, and clear the main list to collect new failures
            retry_queue = failed_links.copy()
            failed_links.clear()
            
            for i, link in retry_queue:
                if downloader.download_item(i, link):
                    success_count += 1
                else:
                    failed_links.append((i, link))
                human_delay(3, 6)
            
        print("\n" + "=" * 60)
        print(f"  Downloaded: {success_count} / {len(links)}")
        if failed_links:
            print(f"  Permanently failed: {len(failed_links)}")
        print(f"  Saved to:   {segment_dir}")
        print("=" * 60)
            
    finally:
        downloader.stop()


if __name__ == "__main__":
    main()
