import pandas as pd
import unicodedata
import re

CHARS_MAP = {
    '\u2018': "'",
    '\u2019': "'",
    '\u201C': '"',
    '\u201D': '"',
    '\u2032': "'",
    '\u2033': '"',
    '\u0092': "'",
    '\u2014': ' - ',
    '\u2013': '-',
    '\u2212': '-',
    '\u2010': '-',
    '\u2011': '-',
    '\u00A0': ' ',
    '\u2009': ' ',
    '\u200A': ' ',
    '\u2005': ' ',
    '\u202F': ' ',
    '\u00D7': 'x',
    '\u00B2': '2',
    '\u2082': '2',
    '\u2211': 'sum',
    '\u2229': ' intersect ',
    '\u222A': ' union ',
    '\u223C': '~',
    '\u2261': '=',
    '\u22C5': '*',
    '\u21CC': '<->',
    '\u2044': '/',
    '\u2026': '...',
    '\u22EF': '...',
    '\u00B7': ' ',
    '\u0430': 'a',
    '\u0435': 'e',
    '\u0438': 'i',
    '\u043F': 'p',
    '\u0440': 'r',
    '\u0441': 'c',
    '\u0443': 'u',
    '\u043D': 'n',
    '\u0434': 'd',
    '\u044F': 'ya',
    '\u041A': 'K',
    '\u0420': 'P',
    '\u0421': 'C',
    '\u20B9': 'Rs.',
    '\u021B': 't',
    '\u03AD': 'e',
    '\u03CC': 'o',
    '\u00A6': '|',
    '\u02DA': '°',
}

CHARS_REMOVE = {
    '\u0080','\u0094','\u00AD','\u0302','\u030A','\u200B','\u2060','\uFEFF',
}

CHARS_WIKIPEDIA = {
    '\u0251','\u0254','\u0259','\u025B','\u026A','\u028A','\u0292','\u02C8','\u02D0','\u0272','\u0282',
}

CHARS_ALLOWED = set(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    ' \t\n'
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    '°±µ'
    'αβγδεζηθικλμνξοπρςστυφχψω'
    'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
    '€£¥$'
    'àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
    'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ'
    'āăąćĉċčďđēĕėęěĝğġģĥħĩīĭįıĳĵķĸĺļľŀłńņňŉŋōŏőœŕŗřśŝşšţťŧũūŭůűųŵŷźżž'
    '©®™§¶†‡~'
)

def clean_text(text):
    s = str(text)

    s = unicodedata.normalize('NFKC', s)

    s = re.sub(r'\s*\(/[^)]*[' + ''.join(CHARS_WIKIPEDIA) + r'][^)]*\)\s*', ' ', s)
    s = re.sub(r'\s*\[[^\]]*[' + ''.join(CHARS_WIKIPEDIA) + r'][^\]]*\]\s*', ' ', s)

    for char, replacement in CHARS_MAP.items():
        if char in s:
            s = s.replace(char, replacement)

    for char in CHARS_REMOVE:
        s = s.replace(char, '')

    for char in CHARS_WIKIPEDIA:
        s = s.replace(char, '')

    s = re.sub(r'\[([A-Z])\]', r'\1', s)
    s = re.sub(r'\[([A-Z]{2,})\]', r'\1', s)
    s = re.sub(r'\[([a-z][^\]]*)\]', r'\1', s)
    s = re.sub(r'\s*\[\.\.\.?\]', '', s)
    s = re.sub(r'\b(CO|CH|H|O|N|SO|NO) (\d)', r'\1\2', s)
    s = s.replace('""', '"')

    s = re.sub(r'[ \t]+', ' ', s)
    s = s.strip()

    return s

def detect_invalid_chars(text):
    invalid_chars = []

    for char in str(text):
        if char not in CHARS_ALLOWED:
            invalid_chars.append(char)

    return invalid_chars

def validate_dataframe(df, text_columns):
    unique_invalid_chars = set()

    for col in text_columns:
        for text in df[col].items():
            for char in detect_invalid_chars(text):
                unique_invalid_chars.add(char)

    if unique_invalid_chars:
        for char in unique_invalid_chars:
            print(f"{char}: {ord(char)}")

def clean_dataframe(df, text_columns):
    df = df.copy()

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    return df