import time
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GDPNewsStreamlit/1.0)",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}


def get_with_retry(url, headers=None, timeout=30, max_retries=3, sleep_s=1.2):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(sleep_s * (attempt + 1))
    raise last_error


def detikcom_url(query, category_id, date_start, date_end, page_num=1, timeout=30, sleep_s=1.2):
    url = (
        "https://www.detik.com/search/searchnews"
        f"?query={query}&siteid={category_id}&sortby=time"
        f"&fromdatex={date_start}&todatex={date_end}"
        f"&page={page_num}&result_type=latest"
    )

    time.sleep(sleep_s)
    response = get_with_retry(
        url,
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        max_retries=3,
        sleep_s=sleep_s,
    )
    return bs(response.content, "html.parser")


def last_page_article_count(query, category_id, date_start, date_end, page_num, timeout=30, sleep_s=1.2):
    last_page = detikcom_url(
        query=query,
        category_id=category_id,
        date_start=date_start,
        date_end=date_end,
        page_num=page_num,
        timeout=timeout,
        sleep_s=sleep_s,
    )

    article_list = last_page.find("div", {"class": "list-content"})
    if not article_list:
        return 0

    articles = article_list.find_all("article", class_="list-content__item")
    return len(articles)


def detikcom_search_results(query, category_id, date_start, date_end, timeout=30, sleep_s=1.2):
    page = detikcom_url(
        query=query,
        category_id=category_id,
        date_start=date_start,
        date_end=date_end,
        timeout=timeout,
        sleep_s=sleep_s,
    )

    pagination = page.find("div", {"class": "pagination"})
    if pagination:
        page_numbers = pagination.find_all("a", class_="pagination__item")
        if page_numbers and len(page_numbers) >= 2:
            try:
                last_page_number = int(page_numbers[-2].text)
            except Exception:
                last_page_number = 1
        else:
            last_page_number = 1

        last_page_articles = last_page_article_count(
            query=query,
            category_id=category_id,
            date_start=date_start,
            date_end=date_end,
            page_num=last_page_number,
            timeout=timeout,
            sleep_s=sleep_s,
        )

        num = (last_page_number - 1) * 10 + last_page_articles
    else:
        article_list = page.find("div", {"class": "list-content"})
        if article_list:
            articles = article_list.find_all("article", class_="list-content__item")
            num = len(articles)
        else:
            num = 0
        last_page_number = 1

    return num, last_page_number


def detikcom_get_content(article_url, timeout=30, sleep_s=1.2):
    def extract_clean_paragraphs(content_page):
        blocked_exact = {
            "",
            "[Gambas:Instagram]",
            "[Gambas:Video 20detik]",
            "\r\nADVERTISEMENT\r\n",
            "\r\n    ADVERTISEMENT\r\n",
            "\r\n    ADVERTISEMENT\r\n  ",
            "\r\n   ADVERTISEMENT\r\n  ",
            "\r\n   ADVERTISEMENT\r\n",
            "\r\n        SCROLL TO RESUME CONTENT\r\n  ",
            "Selengkapnya di halaman selanjutnya.",
            "\n\t\t\t\t\tAyo share cerita pengalaman dan upload photo album travelingmu di sini.\n\n\t\t\t\t\t\t\t\t\t\t\tSilakan Daftar atau Masuk\n",
        }

        paragraphs = []
        for p in content_page.find_all("p"):
            text = p.get_text(" ", strip=True)
            if not text:
                continue
            if text in blocked_exact:
                continue
            if text.startswith("Halaman"):
                continue

            low = text.lower()
            if any(bad in low for bad in [
                "advertisement",
                "scroll to resume content",
                "baca juga",
                "lihat juga",
                "simak video",
                "simak juga video",
                "lihat video",
                "lihat juga video",
            ]):
                continue

            paragraphs.append(text)

        return paragraphs

    try:
        time.sleep(sleep_s)
        response = get_with_retry(
            article_url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            max_retries=3,
            sleep_s=sleep_s,
        )
        content_page = bs(response.content, "html.parser")

        content_list = []

        multiple_page = content_page.find("div", {"class": "detail__multiple"})
        if multiple_page:
            multiple_urls = [x.get("href") for x in multiple_page.find_all("a") if x.get("href")]
            if multiple_urls:
                multiple_urls = multiple_urls[:-1]

            content_list.extend(extract_clean_paragraphs(content_page))

            for page_url in multiple_urls:
                time.sleep(sleep_s)
                page_response = get_with_retry(
                    page_url,
                    headers=DEFAULT_HEADERS,
                    timeout=timeout,
                    max_retries=3,
                    sleep_s=sleep_s,
                )
                sub_page = bs(page_response.content, "html.parser")
                content_list.extend(extract_clean_paragraphs(sub_page))
        else:
            content_list.extend(extract_clean_paragraphs(content_page))

        cleaned = []
        seen_local = set()
        for item in content_list:
            if item not in seen_local:
                cleaned.append(item)
                seen_local.add(item)

        return "\n\n".join(cleaned).strip()

    except Exception:
        return ""


def detikcom_advertorial_check(article_url, timeout=30, sleep_s=1.2):
    try:
        time.sleep(sleep_s)
        response = get_with_retry(
            article_url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            max_retries=2,
            sleep_s=sleep_s,
        )
        content_page = bs(response.content, "html.parser")

        author = content_page.find("meta", {"content": "Advertorial"})
        return bool(author)
    except Exception:
        return False


def scrape_detik_search(
    query,
    siteid,
    from_date,
    to_date,
    max_articles=50,
    timeout=30,
    sleep_s=1.2,
    progress_cb=None,
    include_content=True,
    exclude_advertorial=True,
):
    try:
        results_num, last_page = detikcom_search_results(
            query=query,
            category_id=siteid,
            date_start=from_date,
            date_end=to_date,
            timeout=timeout,
            sleep_s=sleep_s,
        )
    except Exception:
        columns = ["title", "category", "publish_date", "article_url"]
        if include_content:
            columns.append("content")
        return pd.DataFrame(columns=columns)

    article_lists = []
    seen_urls = set()
    seen_titles = set()
    seen_contents = set()

    if results_num == 0:
        columns = ["title", "category", "publish_date", "article_url"]
        if include_content:
            columns.append("content")
        return pd.DataFrame(article_lists, columns=columns)

    pages = last_page

    for i in range(1, pages + 1):
        try:
            page = detikcom_url(
                query=query,
                category_id=siteid,
                date_start=from_date,
                date_end=to_date,
                page_num=i,
                timeout=timeout,
                sleep_s=sleep_s,
            )
        except Exception:
            continue

        articles = page.find_all("article", class_="list-content__item")
        if not articles:
            continue

        for article in articles:
            try:
                link = article.find("a", {"class": "media__link"})
                if not link:
                    continue

                title = link.get("dtr-ttl") or link.get_text(strip=True)
                article_url = link.get("href")

                if not title or not article_url:
                    continue
                if article_url in seen_urls:
                    continue

                subtitle = article.find("h2", class_="media__subtitle")
                category = subtitle.get_text(strip=True) if subtitle else ""

                date_span = article.find("span", title=True)
                publish_date = date_span.get("title") if date_span else ""

                if exclude_advertorial and detikcom_advertorial_check(
                    article_url=article_url,
                    timeout=timeout,
                    sleep_s=sleep_s,
                ):
                    continue

                record = {
                    "title": title,
                    "category": category,
                    "publish_date": publish_date,
                    "article_url": article_url,
                }

                if include_content:
                    content = detikcom_get_content(
                        article_url=article_url,
                        timeout=timeout,
                        sleep_s=sleep_s,
                    )

                    if title in seen_titles or (content and content in seen_contents):
                        continue

                    record["content"] = content
                    if content:
                        seen_contents.add(content)
                else:
                    if title in seen_titles:
                        continue

                article_lists.append(record)
                seen_urls.add(article_url)
                seen_titles.add(title)

                if progress_cb:
                    progress_cb(len(article_lists), max_articles)

                if len(article_lists) >= max_articles:
                    break

            except Exception:
                continue

        if len(article_lists) >= max_articles:
            break

    return pd.DataFrame(article_lists)
