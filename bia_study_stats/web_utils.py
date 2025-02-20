import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def extract_ftp_link(accession_id: str) -> str:
    """
    Extract the FTP link from a BioStudies page for a given accession ID.
    
    Args:
        accession_id: The BioStudies accession ID (e.g. 'S-BIAD1657')
        
    Returns:
        str: The FTP URL for the study data
        
    Raises:
        ValueError: If the FTP link cannot be found or the page cannot be accessed
    """
    base_url = "https://www.ebi.ac.uk/biostudies/BioImages/studies/"
    url = urljoin(base_url, accession_id)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links and look for the FTP one
        for link in soup.find_all('a'):
            if link.get('href', '').startswith('ftp://'):
                return link['href']
            
        raise ValueError(f"Could not find FTP link for {accession_id} in webpage")
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to access page for {accession_id}: {str(e)}")
