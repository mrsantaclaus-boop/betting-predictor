from bs4 import BeautifulSoup

from football.fbref_scraper import FBrefScraper


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# ── Corners ──────────────────────────────────────────────────────────────────

def test_parse_corners_from_team_stats_extra():
    html = """
    <div id="team_stats_extra">
      <div><p>Fouls</p><p>10</p><p>12</p></div>
      <div><p>Corners</p><p>6</p><p>4</p></div>
    </div>
    """
    scraper = FBrefScraper()
    result = scraper._parse_corners_from_soup(_soup(html))
    assert result == {"home": 6, "away": 4}


def test_parse_corners_fallback_to_team_stats_table():
    html = """
    <div id="team_stats">
      <table><tr><td>Corners</td><td>7</td><td>3</td></tr></table>
    </div>
    """
    scraper = FBrefScraper()
    result = scraper._parse_corners_from_soup(_soup(html))
    assert result == {"home": 7, "away": 3}


def test_parse_corners_returns_empty_when_missing():
    html = "<div id='team_stats_extra'><div><p>Fouls</p><p>10</p><p>12</p></div></div>"
    scraper = FBrefScraper()
    assert scraper._parse_corners_from_soup(_soup(html)) == {}


# ── Cards ────────────────────────────────────────────────────────────────────

_CARDS_HTML = """
<table id="stats_abc123_summary">
  <tbody>
    <tr><td data-stat="cards_yellow">1</td><td data-stat="cards_red">0</td></tr>
    <tr><td data-stat="cards_yellow">0</td><td data-stat="cards_red">0</td></tr>
  </tbody>
</table>
<table id="stats_def456_summary">
  <tbody>
    <tr><td data-stat="cards_yellow">2</td><td data-stat="cards_red">1</td></tr>
    <tr><td data-stat="cards_yellow">1</td><td data-stat="cards_red">0</td></tr>
  </tbody>
</table>
"""


def test_parse_cards_sums_yellow_and_red_per_team():
    scraper = FBrefScraper()
    result = scraper._parse_cards_from_soup(_soup(_CARDS_HTML))
    assert result == {"home_yellow": 1, "away_yellow": 3, "home_red": 0, "away_red": 1}


def test_parse_cards_returns_empty_when_tables_missing():
    scraper = FBrefScraper()
    assert scraper._parse_cards_from_soup(_soup("<div>no tables here</div>")) == {}


def test_parse_cards_returns_empty_when_only_one_table():
    html = """
    <table id="stats_abc123_summary">
      <tbody><tr><td data-stat="cards_yellow">1</td><td data-stat="cards_red">0</td></tr></tbody>
    </table>
    """
    scraper = FBrefScraper()
    assert scraper._parse_cards_from_soup(_soup(html)) == {}
