import requests
from bs4 import BeautifulSoup

# Function to parse the Wikipedia page
def parse_wikipedia_page(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers, verify=False)

    if response.status_code != 200:
        print("Failed to retrieve the page")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the page title
    title = soup.find('h1', class_='firstHeading').text.strip()

    # Extract the first non-empty paragraph
    paragraphs = soup.find_all('p')
    first_paragraph = ""
    for p in paragraphs:
        clean_text = p.get_text(strip=True)
        if clean_text:
            first_paragraph = clean_text
            break

    # Extract details from the infobox
    infobox_data = {}
    infobox = soup.find('table', class_='infobox')
    if infobox:
        for row in infobox.find_all('tr'):
            header = row.find('th')
            value = row.find('td')
            if header and value:
                key = header.get_text(strip=True)
                val = value.get_text(strip=True)
                infobox_data[key] = val

    # Extract the full body text
    content_div = soup.find('div', id='mw-content-text')
    full_body_paragraphs = []
    if content_div:
        for p in content_div.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                full_body_paragraphs.append(text)

    full_body_text = "\n".join(full_body_paragraphs)

    return {
        "title": title,
        "first_paragraph": first_paragraph,
        "infobox": infobox_data,
        "full_body_text": full_body_text
    }

# Function to parse and then save the data to a text file
def parse_and_save_wikipedia_page(url, filename):
    data = parse_wikipedia_page(url)
    if data:
        with open(filename, 'w', encoding='utf-8') as f:
            # Write the page title
            f.write(f"Title: {data['title']}\n\n")

            # Write the first paragraph
            f.write("First Paragraph:\n")
            f.write(data['first_paragraph'] + "\n\n")

            # Write the infobox data
            f.write("Infobox Data:\n")
            if data['infobox']:
                for key, val in data['infobox'].items():
                    f.write(f"{key}: {val}\n")
            else:
                f.write("No infobox data found.\n")

            f.write("\n")

            # Write the full body text
            f.write("Full Body Text:\n")
            f.write(data['full_body_text'] + "\n")
    else:
        print("No data to save.")

if __name__ == "__main__":
    url = 'https://ru.wikipedia.org/wiki/Университет_ИТМО'
    output_file = 'itmo_wiki_data.txt'
    parse_and_save_wikipedia_page(url, output_file)
    print(f"Data saved to {output_file}")