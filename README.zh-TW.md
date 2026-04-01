# ChromoLab Ops Portal

ChromoLab 是一套面向研究團隊的 AI 營運後台。

它以染色體影像為示範案例，但真正要解決的問題，不只是「把模型跑起來」，而是把資料標註、模型訓練、驗證比較、版本追蹤與下一步研究方向，收進同一套可持續運轉的流程。

English version: [README.md](README.md)

## 這個專案想做的事

- 讓 `admin`、`annotator`、`viewer` 有清楚的分工邊界
- 讓研究人員可以上傳 YOLO 格式資料集並快速盤點成熟度
- 讓標註人員直接在線上修正 segmentation polygon
- 讓管理者排模型比較、控制訓練條件、追蹤 revision
- 讓系統依照資料變化持續做優化與回看
- 讓結果不只是一個分數，而是能回推出下一步研究方向

## 平台定位

這不是單次 demo，也不是單一模型頁面。

ChromoLab 比較像一個 AI 研究中控台：

1. 新資料進來後，先整理資料批次與標註缺口
2. 標註員在線上補齊 revision
3. 管理者決定模型比較與持續優化策略
4. 系統留下 metrics、preview、log 與版本軌跡
5. 研究團隊根據結果決定下一輪資料與實驗方向

## 核心能力

- 分權限後台
- YOLO zip 資料集匯入
- 線上標註
- 模型比較排程
- Continuous optimizer
- 訓練 preview 與 log 回看
- 舊版 inference gallery 共存
- 研究方向導航建議

## 技術堆疊

- FastAPI
- Uvicorn
- Ultralytics YOLO
- Vanilla HTML / CSS / JS

## 專案結構

- `app/`: API、權限、資料管理、job worker、optimizer、前端頁面
- `config/`: 預設帳號與種子設定
- `scripts/`: 訓練、部署、smoke test、tunnel 與啟停腳本
- `requirements.txt`: Python 相依套件

## 本機啟動

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

啟動後：

- `/` 是英文介紹頁
- `/zh-TW` 是中文版介紹頁
- `/portal` 是後台登入頁

## 預設帳號

- `admin / admin1234`
- `annotator / annotator1234`
- `viewer / viewer1234`

正式環境請先改密碼。
