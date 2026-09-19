import httpx

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def search_pubmed(query: str, retmax: int = 15) -> list[str]:
    response = httpx.get(
        f"{BASE_URL}esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax},
    )
    response.raise_for_status()
    return response.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list[str]) -> str:
    response = httpx.get(
        f"{BASE_URL}efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "text"},
    )
    response.raise_for_status()
    return response.text


if __name__ == "__main__":
    pmids = search_pubmed("SSVEP brain computer interface")
    print(f"{len(pmids)} makale bulundu: {pmids}")
    text = fetch_abstracts(pmids)
    print(text[:1500])