from pathlib import Path

from .utils.strenum import StrEnum

# Style-Bert-VITS2 のバージョン
VERSION = "2.4.1"

BASE_DIR = Path(__file__).parent


# 利用可能な言語
## JP-Extra モデル利用時は JP 以外の言語の音声合成はできない
class Languages(StrEnum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


# 言語ごとのデフォルトの BERT トークナイザーのリポジトリ
DEFAULT_BERT_TOKENIZER_REPOS = {
    Languages.JP: "ku-nlp/deberta-v2-large-japanese-char-wwm",
    Languages.EN: "microsoft/deberta-v3-large",
    Languages.ZH: "hfl/chinese-roberta-wwm-ext-large",
}

# デフォルトのユーザー辞書ディレクトリ
## .nlp.japanese.user_dict モジュールのデフォルト値として利用される
## ライブラリとしての利用などで外部のユーザー辞書を指定したい場合は、user_dict 以下の各関数の実行時、引数に辞書データファイルのパスを指定する
DEFAULT_USER_DICT_DIR = BASE_DIR / "dict_data"

# デフォルトの推論パラメータ
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 5.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 0.7
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0

# Gradio のテーマ
## Built-in theme: "default", "base", "monochrome", "soft", "glass"
## See https://huggingface.co/spaces/gradio/theme-gallery for more themes
GRADIO_THEME = "derekzen/stardust"
