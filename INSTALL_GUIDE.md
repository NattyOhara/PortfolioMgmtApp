# インストールガイド

Gemini API + Google Search API機能を使用するために必要なライブラリのインストール手順です。

## 1. 仮想環境の確認

まず、正しい仮想環境で作業していることを確認してください：

### Windowsの場合：
```cmd
# 仮想環境をアクティベート
venv\Scripts\activate

# Pythonのパスを確認
where python
# 出力例: C:\Users\...\PortfolioManagementApp\venv\Scripts\python.exe

# 現在インストールされているパッケージを確認
pip list
```

### Linux/Macの場合：
```bash
# 仮想環境をアクティベート
source venv/bin/activate

# Pythonのパスを確認
which python
# 出力例: /path/to/PortfolioManagementApp/venv/bin/python

# 現在インストールされているパッケージを確認
pip list
```

## 2. 必要なライブラリのインストール

仮想環境がアクティブな状態で、以下のコマンドを順番に実行してください：

```bash
# 1. pipを最新版にアップデート
pip install --upgrade pip

# 2. Google Generative AI (Gemini API用)
pip install google-generativeai

# 3. Google API Python Client (Google Search API用)
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

# 4. スクレイピング用ライブラリ
pip install beautifulsoup4 requests

# 5. すべての依存関係を再インストール
pip install -r requirements.txt --upgrade
```

## 3. インストールの確認

インストールが成功したかを確認してください：

```bash
# 各ライブラリのインポートテスト
python -c "import google.generativeai; print('✓ google-generativeai OK')"
python -c "from googleapiclient.discovery import build; print('✓ google-api-python-client OK')"
python -c "from bs4 import BeautifulSoup; print('✓ beautifulsoup4 OK')"
python -c "import requests; print('✓ requests OK')"
```

すべて「OK」と表示されれば成功です。

## 4. .envファイルの設定

以下の環境変数を.envファイルに設定してください：

```env
# Google Cloud PlatformのAPIキー
GOOGLE_API_KEY=AIzaSyBx1234567890abcdefghijklmnopqrstuv

# Programmable Search EngineのID
GOOGLE_SEARCH_ENGINE_ID=017643444788069204610:4gvhea_mvga

# Google AI StudioのAPIキー
GEMINI_API_KEY=AIzaSyC-9876543210zyxwvutsrqponmlkjihgfe
```

## 5. Streamlitアプリの起動

```bash
streamlit run app.py
```

## トラブルシューティング

### エラー: "No module named 'google'"
- 仮想環境が正しくアクティベートされているか確認
- `pip install google-generativeai google-api-python-client` を再実行

### エラー: "ImportError: cannot import name 'build'"
- `pip install --upgrade google-api-python-client` を実行

### エラー: "ModuleNotFoundError: No module named 'googleapiclient'"
- `pip install google-api-python-client` を実行

### パッケージのバージョン競合が発生した場合：
```bash
# 既存のGoogleライブラリを削除して再インストール
pip uninstall google-generativeai google-api-python-client google-auth-httplib2 google-auth-oauthlib -y
pip install google-generativeai google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## 6. 動作確認

アプリが起動したら：
1. 「運用レポート」タブに移動
2. エラーメッセージが表示されないことを確認
3. 「運用レポートを生成（ニュース分析付き）」ボタンが表示されることを確認

これで、Gemini API + Google Search APIを使用した新機能が利用可能になります。