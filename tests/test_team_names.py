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
