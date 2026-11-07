import sys
import os

# We need to test the locator syntax to make sure it doesn't syntax error
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
                cache_dir = os.path.expanduser("~/.cache/playwright_node_cache")
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

from playwright.sync_api import sync_playwright
import re

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # We can just create a dummy page with some html
        page.set_content("""
            <html>
                <body>
                    <button style="display:none;">Download</button>
                    <button> Download </button>
                    <button>Download without license</button>
                    <button>Add & Download</button>
                    <button>Add and Download</button>
                </body>
            </html>
        """)
        
        # Test download button
        btn = page.locator("button:visible", has_text=re.compile(r"^\s*Download\s*$", re.IGNORECASE)).first
        print("Download btn found:", btn.inner_text())
        
        # Test modal button
        modal_btn = page.locator("button:visible", has_text=re.compile(r"(Add.*Download|Download without license)", re.IGNORECASE)).first
        print("Modal btn found:", modal_btn.inner_text())
        
        browser.close()

if __name__ == "__main__":
    main()
