import contractions
import emoji
from ftfy import fix_text as fix_text
import re
from sacremoses import MosesPunctNormalizer
import sys
import unicodedata

from typing import Callable, Literal, Optional


class String(object):
    """
    A class for processing strings using a series of sequential text processing tasks.
    """

    def __init__(self, text: str):
        self.text = str(text)
    
    def __str__(self) -> str:
        return self.to_string()
    
    def to_string(self, **kwargs):
        return self.text

    def __getattr__(self, func):
        func = getattr(sys.modules[__name__], func)

        def wrapper(*args, **kwargs):
            self.text = func(self.text, *args, **kwargs)

            return self
        
        return wrapper


def process(text: str, split: Literal['lines', 'words'] = 'lines', tasks: list[str] = None) -> str:
    """
    Process the text using a series of sequential text preprocessing tasks.

    Parameters
    ----------
    text : str
        The input text.
    split : str, optional
        The method to use for splitting the text. Default is 'lines'.
        If 'lines', the text is split into lines.
        If 'words', the text is split into words.
    tasks : list, optional
        A list of text preprocessing tasks to perform.
        If not provided, the default tasks are used.

    Returns
    -------
    str
        The processed text.
    """
    if not isinstance(text, str):
        return

    text = str(text)

    if tasks is None:
        return normalize_whitespace(text)

    chunks = split(text, method=split)

    for i, chunk in enumerate(chunks):
        for task in tasks:
            params = {}
            if isinstance(task, (list, tuple)):
                task, params = task

            if callable(task):
                fn = task
            elif isinstance(task, str):
                fn = getattr(sys.modules[__name__], task)
            else:
                continue

            chunk = fn(chunk, **params)

        chunks[i] = chunk
    
    text = join(chunks, method=split)

    return normalize_whitespace(text)


def split(text: str, method: str | Callable = 'lines') -> list:
    """
    Split the text into chunks using a specified method.

    Parameters
    ----------
    text : str
        The input text.
    method : str or callable, optional
        The method to use for splitting the text. Default is 'lines'.
        If a string is provided, the text is split based on the specified method.
        If a callable is provided, it is used to split the text.

    Returns
    -------
    list
        A list of text chunks.
    """
    if not isinstance(text, str):
        return

    text = str(text)

    if method == 'lines':
        return text.splitlines()
    elif method == 'words':
        return text.split()
    elif callable(method):
        return method(text)
    else:
        return [text]


def join(chunks: list, method: str | Callable = 'lines') -> str:
    """
    Join the chunks into a single text using a specified method.

    Parameters
    ----------
    chunks : list
        A list of text chunks.
    method : str or callable, optional
        The method to use for joining the chunks. Default is 'lines'.
        If a string is provided, the chunks are joined based on the specified method.
        If a callable is provided, it is used to join the chunks.

    Returns
    -------
    str
        The joined text.
    """
    if not isinstance(chunks, (list, tuple)):
        return

    if method == 'lines':
        return '\n'.join(chunks)
    elif method == 'words':
        return ' '.join(chunks)
    elif callable(method):
        return method(chunks)
    else:
        return chunks[0]


def is_string(x):
    """
    Check if the input is a string.

    Parameters
    ----------
    x : any
        The input to check.

    Returns
    -------
    bool
        Whether the input is, indeed, a string.
    """
    if x is None:
        return False

    return isinstance(x, str)


def clean_string(text: str) -> str:
    """
    Clean the text by removing leading/trailing whitespace and normalizing whitespace.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    """
    if not is_string(text):
        return

    return normalize_whitespace(
        normalize_unicode(text)
    )


def replace_chars(text: str, chars: dict = None) -> str:
    """
    Replace special characters with their corresponding alternative.

    Parameters
    ----------
    text : str
        The input text.
    chars : dict, optional
        A dictionary of special characters and their alternatives.
        If not provided, the default dictionary is used.

    Returns
    -------
    str
        The processed text.
    """
    if chars is None:
        chars = {
            '¨': '"',
            '«': '"',
            '»': '"',
            '´': "'",
            '‘': "'",
            '’': "'",
            '“': '"',
            '”': '"',
            '·': '.',
            '…': '...'
        }

    for char, replace in chars.items():
        text = text.replace(char, replace)

    return text


def normalize_unicode(text: str, method: Literal['NFC', 'NFD', 'NFKC', 'NFKD'] = 'NFC') -> str:
    """
    Normalize Unicode characters using a specific normalization method.

    Parameters
    ----------
    text : str
        The input text.
    method : str, optional
        The normalization method to use. Default is 'NFC'.

    Returns
    -------
    str
        The processed text.
    """
    return fix_text(
        replace_chars(str(text)),
        normalization=method
    )


def remove_accents(text: str) -> str:
    """
    Remove accents from characters, replacing them with their Unicode base form.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    """
    return ''.join(
        c for c in normalize_unicode(text, method='NFKD') if not unicodedata.combining(c)
    )


def to_lower(text: str) -> str:
    """
    Return a lowercase version of the string.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    """
    return text.lower()


def to_upper(text: str) -> str:
    """
    Return an uppercase version of the string.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    """
    return text.upper()


def to_title(text: str) -> str:
    """
    Return a titlecased version of the string where words start with an uppercase character and
    the remaining characters are lowercase.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> to_title("hello world")
    "Hello World"
    """
    return text.title()


def to_camel(text: str) -> str:
    """
    Return a camelcased version of the string where words are concatenated without spaces and
    each word starts with an uppercase character.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> to_camel("hello world")
    "HelloWorld"
    """
    return re.sub(
        r'[\s_-]+',
        '',
        re.sub(
            r'[^a-zA-Z0-9\s_-]',
            '',
            remove_accents(text.title())
        )
    ).strip()


def to_snake(text: str) -> str:
    """
    Return a snakecased version of the string where words are concatenated with an underscore and
    each word is lowercase.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> to_snake("Hello world")
    "hello_world"
    """
    return re.sub(
        r'[\s_-]+',
        '_',
        re.sub(
            r'[^a-zA-Z0-9\s_-]',
            '',
            remove_accents(text.lower())
        )
    ).strip('_')


def to_slug(text: str, separator: str = '-') -> str:
    """
    Return a slugified version of the string where words are concatenated with a separator and
    non-alphabetic characters are removed.

    Parameters
    ----------
    text : str
        The input text.
    separator : str, optional
        The separator to use for the slug. Default is '-'.

    Returns
    -------
    str
        The slugified text.
    
    Examples
    --------
    >>> to_slug("Hello world", separator='-')
    "hello-world"
    """
    return re.sub(
        r'[\s_-]+',
        separator,
        re.sub(
            r'[^a-zA-Z0-9\s_-]',
            '',
            remove_accents(text.lower())
        )
    ).strip(separator)


def limit_charseq(text: str, limit: int = 3) -> str:
    """
    Limit the number of consecutive occurrences of a character.

    Parameters
    ----------
    text : str
        The input text.
    limit : int, optional
        The maximum number of consecutive occurrences. Default is 3.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> limit_charseq("Hellooooo world!!!!! I'm back...", limit=3)
    "Hello world! I'm back..."
    """
    return re.sub(
        r'(.)' + r'\1' * (limit - 1) + '+',
        r'\1' * limit,
        text
    )


def remove_html(text: str, replace: str = '') -> str:
    """
    Remove HTML tags from the text.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the HTML tags with. Default is an empty string.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_html("<p>Hello world</p>")
    "Hello world"
    >>> remove_html("Hello <a href="http://world.com">world</a>!")
    "Hello world!"
    """
    patterns = [
        (r'<\/p>\s*<p>', '</p>\n<p>'),
        (r'(\s*<br[\s\/]*>\s*)', r'\1\n'),
        (r'&nbsp;', ' '),
        (r'<.*?>', replace)
    ]

    for (pattern, replace) in patterns:
        text = re.sub(pattern, replace, text)

    return normalize_whitespace(text)


def remove_urls(text: str, replace: str = '') -> str:
    """
    Remove URLs from the text.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the URLs with. Default is an empty string.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_urls("Visit https://www.world.com for more information.")
    "Visit for more information."
    >>> remove_urls("Visit https://www.world.com for more information.", replace="[URL]")
    "Visit [URL] for more information."
    """
    return normalize_whitespace(
        re.sub(
            r'(https?:/{2})?\w+(\.(\w{2,})){1,3}([/\w\-_]+)?\??[\w\-_&=]*',
            replace,
            text
        )
    )


def remove_emails(text: str, replace: str = '') -> str:
    """
    Remove email addresses from the text.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the email addresses with. Default is an empty string.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_emails("Contact us at contact@world.com for information.")
    "Contact us at for information."
    >>> remove_emails("Contact us at contact@world.com for information.", replace="[EMAIL]")
    "Contact us at [EMAIL] for information."
    """
    return normalize_whitespace(
        re.sub(
            r'(?:^|(?<=\b))[\w.\-_]+@\w+(\.(\w{2,})){1,3}(?:$|(?=\b))',
            replace,
            text
        )
    )


def remove_hashtags(text: str, replace: str = '', delimiters: Optional[str | tuple[str, str]] = None) -> str:
    """
    Remove hashtags from the text.
    Hashtags are words or phrases prefixed with the '#' symbol, often used on social media platforms.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the hashtags with. Default is an empty string.
    delimiters : str or tuple, optional
        A pair of delimiters to wrap the hashtag texts. Default is None.
        If a string is provided, it is used for both the opening and closing characters.
        If a tuple is provided, the first item is the opening character and the second item is the closing one.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_hashtags("Hello #world")
    "Hello world"
    >>> remove_hashtags("Hello #world!", replace="[HASHTAG]")
    "Hello [HASHTAG]"
    >>> remove_hashtags("Hello #world!", delimiters="|")
    "Hello |world|"
    >>> remove_hashtags("Hello #world!", delimiters=["[", "]"])
    "Hello [world]"
    """
    if delimiters is not None:
        if isinstance(delimiters, str):
            delimiters = (delimiters, delimiters)

        return re.sub(
            r'#(\w*)',
            lambda x: delimiters[0] + re.sub(
                r'([A-Z]+)',
                r' \1',
                x.groups()[0]
            ).strip() + delimiters[1],
            text
        )

    return normalize_whitespace(
        re.sub(r'#\w*', replace, text)
    )


def remove_mentions(text: str, replace: str = '', delimiters: Optional[str | tuple[str, str]] = None) -> str:
    """
    Remove mentions from the text.
    Mentions are usernames or handles prefixed with the '@' symbol, often used on social media platforms.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the mentions with. Default is an empty string.
    delimiters : str or tuple, optional
        A pair of delimiters to wrap the mention texts. Default is None.
        If a string is provided, it is used for both the opening and closing characters.
        If a tuple is provided, the first item is the opening character and the second item is the closing one.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_mentions("This is @world talking.")
    "This is world talking."
    >>> remove_mentions("This is @world talking.", replace="[USER]")
    "This is [USER] talking."
    >>> remove_mentions("This is @world talking.", delimiters="|")
    "This is |world| talking."
    >>> remove_mentions("This is @world talking.", delimiters=["[", "]"])
    "This is [world] talking."
    """
    if delimiters is not None:
        if isinstance(delimiters, str):
            delimiters = (delimiters, delimiters)

        return re.sub(
            r'@(\w*)',
            lambda x: delimiters[0] + re.sub(
                r'([A-Z]+)',
                r' \1',
                x.groups()[0]
            ).strip() + delimiters[1],
            text
        )

    return normalize_whitespace(
        re.sub(r'@\w*', replace, text)
    )


def remove_emojis(text: str, replace: str = '', delimiters: Optional[str | tuple[str, str]] = None, lang: Optional[str] = None) -> str:
    """
    Remove emojis from the text.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the emojis with. Default is an empty string.
    delimiters : str or tuple, optional
        A pair of delimiters to wrap the emoji replacements. Default is None.
        If a string is provided, it is used for both the opening and closing characters.
        If a tuple is provided, the first item is the opening character and the second item is the closing one.
    lang : str, optional
        The language to use for the emoji replacements. Default is None.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_emojis("Hello 🤔")
    "Hello"
    >>> remove_emojis("Hello 🤔", replace="[EMOJI]")
    "Hello [EMOJI]"
    >>> remove_emojis("Hello 🤔", delimiters="|")
    "Hello |thinking|"
    >>> remove_emojis("Hello 🤔", delimiters=["[", "]"], lang="en")
    "Hello [thinking]"
    """
    if delimiters is not None:
        if isinstance(delimiters, str):
            delimiters = (delimiters, delimiters)

        return emoji.demojize(text, language=lang, delimiters=delimiters)

    return normalize_whitespace(
        emoji.replace_emoji(text, replace=replace)
    )


def remove_emoticons(text: str, replace: str = '') -> str:
    """
    Remove emoticons from the text.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the emoticons with. Default is an empty string.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_emoticons("Hello :-)")
    "Hello"
    >>> remove_emoticons("Hello :-)", replace="[EMOTICON]")
    "Hello [EMOTICON]"
    """
    return normalize_whitespace(
        re.sub(
            r'[:;=8][-o*\']?[)\]}(\[{\\|@*dpo]|\^[_-]?[\^o]|<[/\\]?3|\bx+[dpo]*\b',
            replace, text, flags=re.IGNORECASE
        )
    )


def remove_laughters(text: str, replace: str = '') -> str:
    """
    Remove laughters from the text.

    Parameters
    ----------
    text : str
        The input text.
    replace : str, optional
        The string to replace the laughters with. Default is an empty string.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_laughters("Hello hahaha")
    "Hello"
    >>> remove_laughters("Hello hahaha", replace="[LAUGHTER]")
    "Hello [LAUGHTER]"
    """
    return normalize_whitespace(
        re.sub(
            r'\b[jhae][jhae]+[ae][jh][jhae]+\b',
            replace,
            text,
            flags=re.IGNORECASE
        )
    )


def fix_contractions(text: str) -> str:
    """
    Fix contractions, expanding them to their full form.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> fix_contractions("I'm going to the store.")
    "I am going to the store."
    >>> fix_contractions("You're the best!")
    "You are the best!"
    """
    return contractions.fix(text)


def fix_punctuation(text: str) -> str:
    """
    Normalize punctuation in the text.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> fix_punctuation("Hello, world!!!!!")
    "Hello, world!"
    >>> fix_punctuation("´Hello World´ - she said…")
    "'Hello World' - she said..."
    >>> fix_punctuation("This is version 2.0 of this software.")
    "This is version 2.0 of this software."
    """
    text = MosesPunctNormalizer().normalize(text)

    punct_pre = r'¿¡\(\['
    punct_pos = r';:\?\!\)\]'

    patterns = [
        (r'\s?(?<!\d)([\.,]+)|([\.,]+)(?!\d)\s?', r'\1\2 '),  # Normalize non-decimal points/commas
        (r'\s?([{}]+)\s?'.format(punct_pre), r' \1'),  # Normalize puncts (pre)
        (r'\s?([{}]+)\s?'.format(punct_pos), r'\1 '),  # Normalize puncts (pos)
        (r'(\.*[,{}]){{2,}}'.format(punct_pos + punct_pre), r'\1'),  # Remove repeated sequences of non-dot punct
        (r'\.{2,}', '...')  # Limit to 3 consecutive dots
    ]

    for (pattern, replace) in patterns:
        text = re.sub(pattern, replace, text)

    return normalize_whitespace(text)


def remove_eol_chars(text: str) -> str:
    """
    Remove end-of-line characters from the text.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> remove_eol_chars("Hello\n world\r\n")
    "Hello world"
    """
    return re.sub(
        r'((\r\n)|[\n\v])+',
        '\n',
        text
    ).replace('\n', ' ')


def remove_thread_info(text: str) -> str:
    """
    Remove thread information from the text.
    This applies to email and forum threads, where the original message is often included in the reply
    and does not provide any additional value.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    """
    patterns = [
        r'-+(Original|Forwarded) [Mm]essage-+\s*From:',
        r'On \w{3}, .* at .* wrote:',
        r'From:.*[<\[].*[>\]]'
    ]

    for pattern in patterns:
        text = re.split(pattern, text)[0]

    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in the text.
    This includes removing leading/trailing whitespace and replacing multiple spaces with a single space.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The processed text.
    """
    return re.sub(
        r'(?!\n)\s+',
        ' ',
        re.sub(
            r'((\r\n)|[\n\v])+',
            '\n',
            text
        )
    ).strip()


def truncate(text: str, limit: int = 256, replace: str = '', unit: Literal['bytes', 'chars'] = 'bytes', encoding: str = 'utf-8') -> str:
    """
    Truncate the text to a specified length.

    Parameters
    ----------
    text : str
        The input text.
    limit : int, optional
        The maximum length of the text. Default is 256.
    replace : str, optional
        The string to replace the truncated text with. Default is an empty string.
    unit : str, optional
        The unit of the limit. Default is 'bytes'.
    encoding : str, optional
        The encoding to use for the text. Default is 'utf-8'.

    Returns
    -------
    str
        The processed text.
    
    Examples
    --------
    >>> truncate("Hello world", limit=5)
    "Hello"
    >>> truncate("Hello world", limit=5, replace="...")
    "Hello..."
    """
    limit -= len(replace)

    if unit == 'bytes':
        btext = text.encode(encoding)

        if len(btext) <= limit:
            return text

        i = limit
        while i > 0 and (btext[i] & 0xC0) == 0x80:
            i -= 1

        text = btext[:i].decode(encoding) + replace
    else:
        if len(text) <= limit:
            return text

        text = text[:limit] + replace

    return normalize_whitespace(text)
