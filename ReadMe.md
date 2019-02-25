1. `__init__.py`を`anaconda3/lib/python3.6/site-packages/gym/envs`フォルダへコピーする（オリジナルの名前を変更しておく）。
2. `cartpole.py`を`~/anaconda3/lib/python3.6/site-packages/gym/envs/classic_control`フォルダへコピーする（オリジナルの名前を変更しておく）。
3. `core.py`を`/Users/me/anaconda3/lib/python3.6/site-packages/rl`フォルダへコピーする（オリジナルの名前を変更しておく）。
4. `dqn_cartpole.py`を`ホーム`フォルダへコピーする（オリジナルの名前を変更しておく）。
5. `cartpole_results`を`ホーム`フォルダに作成する（オリジナルがあれば，名前を変更しておく）。  
    ※）このフォルダに学習結果のCSVファイルが格納される。
6. 実行はターミナル（Windowsではcmd）を開き，
```
$ python3 dqn_cartpole.py
```
を入力して［return］（Windowsでは［Enter］）キーを押す。
※）$ はプロンプトなので入力してはいけない。
