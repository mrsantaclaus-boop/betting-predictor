from football.team_names import keywords, teams_match, match_score


def test_inter_variants_match():
    assert teams_match("FC Internazionale Milano", "Inter Milan")
    assert teams_match("FC Internazionale Milano", "Internazionale")
    assert teams_match("Internazionale", "Inter")


def test_milan_variants_match():
    assert teams_match("AC Milan", "Milan")
    assert teams_match("AC Milan", "AC Milan")


def test_milan_derby_does_not_collide():
    assert not teams_match("FC Internazionale Milano", "AC Milan")
    assert not teams_match("Inter Milan", "AC Milan")
    assert not teams_match("AC Milan", "Inter")


def test_stopwords_alone_never_match():
    assert not teams_match("FC", "AC")
    assert not teams_match("", "AC Milan")


def test_generic_words_are_dropped():
    assert keywords("US Lecce") == {"lecce"}
    assert keywords("Parma Calcio 1913") == {"parma"}
    assert keywords("Bologna FC 1909") == {"bologna"}
    assert keywords("Hellas Verona FC") == {"verona"}


def test_match_score_prefers_fuller_overlap():
    assert match_score("Atletico Madrid", "Atletico Madrid") > match_score("Atletico Madrid", "Real Madrid")
    assert match_score("Real Madrid", "Real Madrid CF") >= 2


# ── La Liga: football-data.org name -> ESPN name, one to one ──────────────────
# Real names fetched 2026-11-19 (standings PD / ESPN esp.1 teams).
_LALIGA = {
    "FC Barcelona": "Barcelona",
    "Real Madrid CF": "Real Madrid",
    "Club Atlético de Madrid": "Atlético Madrid",
    "Real Betis Balompié": "Real Betis",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Sevilla FC": "Sevilla",
    "Athletic Club": "Athletic Club",
    "Villarreal CF": "Villarreal",
    "RC Deportivo La Coruña": "Deportivo",
    "Real Racing Club de Santander": "Racing Santander",
    "RC Celta de Vigo": "Celta Vigo",
    "Deportivo Alavés": "Alavés",
    "RCD Espanyol de Barcelona": "Espanyol",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Getafe CF": "Getafe",
    "CA Osasuna": "Osasuna",
    "Levante UD": "Levante",
    "Elche CF": "Elche",
    "Valencia CF": "Valencia",
    "Málaga CF": "Málaga",
}


def test_laliga_every_club_matches_its_own_espn_name_uniquely():
    espn = list(_LALIGA.values())
    for fd_name, espn_name in _LALIGA.items():
        scores = sorted(((match_score(fd_name, e), e) for e in espn), reverse=True)
        assert scores[0][1] == espn_name, (fd_name, scores[:3])
        assert scores[0][0] > scores[1][0], ("tie", fd_name, scores[:3])
