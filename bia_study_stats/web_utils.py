import re
import time
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def extract_ftp_link(accession_id: str) -> str:
    """
    Extract the FTP link from a BioStudies page for a given accession ID.
    Uses Selenium to handle JavaScript-rendered content.
    
    Args:
        accession_id: The BioStudies accession ID (e.g. 'S-BIAD1657')
        
    Returns:
        str: The FTP URL for the study data
        
    Raises:
        ValueError: If the FTP link cannot be found or the page cannot be accessed
    """
    base_url = "https://www.ebi.ac.uk/biostudies/BioImages/studies/"
    url = urljoin(base_url, accession_id)
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Wait for the page to load and JavaScript to execute
        time.sleep(2)  # Give JavaScript time to run
        
        # Look for FTP links
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href")
            if href and href.startswith("ftp://"):
                driver.quit()
                return href
        
        driver.quit()
        raise ValueError(f"Could not find FTP link for {accession_id} in webpage")
        
    except Exception as e:
        if 'driver' in locals():
            driver.quit()
        raise ValueError(f"Failed to access page for {accession_id}: {str(e)}")
