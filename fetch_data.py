import json
import requests
from html.parser import HTMLParser


class AtCoderCSRFExtractor(HTMLParser):
    def __init__(self):
        super(AtCoderCSRFExtractor, self).__init__()
        self.csrf = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name") == "csrf_token":
            self.csrf = attrs["value"]

    def extract(self, html):
        self.feed(html)
        if self.csrf is not None:
            return self.csrf
        else:
            raise ValueError("Failed to extract CSRF token")


def login():
    with open("secret.json") as f:
        secret = json.load(f)
    session = requests.Session()
    get_response = session.get("https://atcoder.jp/login")
    extractor = AtCoderCSRFExtractor()
    csrf = extractor.extract(get_response.text)
    form_values = {"username": secret["atcoder_user"], "password": secret["atcoder_pass"], "csrf_token": csrf}
    post_response = session.post("https://atcoder.jp/login", data=form_values)
    if post_response.status_code != 200:
        raise Exception(str(post_response))
    return session


session = login()


for contest_num in range(272, 283):
    contest_id = f"abc{contest_num}"
    print(contest_id)
    results = session.get(f"https://atcoder.jp/contests/{contest_id}/standings/json").json()
    with open(f"data/{contest_id}.json", "w") as f:
        json.dump(results, f)
