from hankel._bases import _affixes_of, by_all_affixes


def test_finds_all_prefixes():
    assert (sorted(_affixes_of([tuple('abc'), tuple('bcd'), tuple('z')], prefix=True))
            == list(map(tuple, ['', 'a', 'ab', 'abc', 'b', 'bc', 'bcd', 'z'])))


def test_finds_all_suffixes():
    assert (sorted(_affixes_of([tuple('abc'), tuple('bcd'), tuple('z')], prefix=False))
            == list(map(tuple, ['', 'abc', 'bc', 'bcd', 'c', 'cd', 'd', 'z'])))


def test_finds_basis_with_all_affixes():
    prefs, suffs = by_all_affixes([tuple('abc'), tuple('bcd'), tuple('z')])

    assert prefs == tuple(map(tuple, ('', 'a', 'ab', 'abc', 'b', 'bc', 'bcd', 'z',
                                      'a', 'aa', 'aba', 'abca', 'ba', 'bca', 'bcda', 'za',
                                      'b', 'ab', 'abb', 'abcb', 'bb', 'bcb', 'bcdb', 'zb',
                                      'c', 'ac', 'abc', 'abcc', 'bc', 'bcc', 'bcdc', 'zc',
                                      'd', 'ad', 'abd', 'abcd', 'bd', 'bcd', 'bcdd', 'zd',
                                      'z', 'az', 'abz', 'abcz', 'bz', 'bcz', 'bcdz', 'zz')))

    assert suffs == tuple(map(tuple, ('', 'abc', 'bc', 'bcd', 'c', 'cd', 'd', 'z')))

 