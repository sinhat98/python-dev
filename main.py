from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import sys
import logging
from logging.handlers import RotatingFileHandler
from pprint import pprint
import re
import time

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ストリームハンドラ（標準出力）
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)  # INFOレベルのログを標準出力に
stdout_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# # ストリームハンドラ（標準エラー出力）
# stderr_handler = logging.StreamHandler(sys.stderr)
# stderr_handler.setLevel(logging.WARNING)  # WARNINGレベル以上のログを標準エラー出力に
# stderr_handler.setFormatter(
#     logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# )

# ロガーにハンドラを追加
logger.addHandler(stdout_handler)
# logger.addHandler(stderr_handler)

# # ファイル出力のためのハンドラ
# file_handler = RotatingFileHandler("app.log", maxBytes=10000, backupCount=3)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

app = FastAPI()


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    # logger.debug("Request body: %s", body["queryResult"])

    log_dict(body["queryResult"])

    intent_name = body["queryResult"]["intent"]["displayName"]

    if intent_name == "reservation_hour_intent":
        logger.info("Handling reservation time intent")
        return JSONResponse(content=handle_reservation_time(body))
    elif intent_name == "response_final":
        logger.info("Handling final response intent")
        return JSONResponse(content=handle_filnal_response(body))
    else:
        logger.info("Received an unsupported intent: %s", intent_name)
        return JSONResponse(
            content={"fulfillmentText": "対応していないインテントです。"}
        )


def handle_filnal_response(body):
    log_dict(body["queryResult"])
    payload = body["queryResult"]["fulfillmentMessages"][1]["payload"]
    model = payload.get("model")

    logger.debug("Payload: %s", payload)
    if payload["use_vad"]:
        if model is not None:
            logger.info("Using VAD with model: %s", model)
        else:
            logger.info("Using VAD with rule base")
    else:
        time.sleep(3)
    fulfillment_text = "ありがとうございました。"
    return {"fulfillmentText": fulfillment_text}


def handle_reservation_time(body):
    parameters = body["queryResult"]["parameters"]
    reservation_time = parameters.get("time")
    hour = re.search(r"\b\d{1,2}\b", reservation_time)
    if hour is None:
        fulfillment_text = "0~24時の時間を指定してください。"
        logger.error("Invalid time format: %s", hour)
        return {"fulfillmentText": fulfillment_text}
    hour = int(hour.group(0))
    logger.debug("Reservation time: %s", hour)
    try:

        if not (0 <= hour <= 24):
            raise ValueError("時間は0から24の間で指定してください。")
    except ValueError as e:
        fulfillment_text = str(e)
        logger.error("Invalid time format or value: %s", reservation_time)
        return {"fulfillmentText": fulfillment_text}

    if reservation_time:
        if 9 <= hour < 19:
            fulfillment_text = "予約時間は承知しました。"
        else:
            fulfillment_text = "申し訳ありませんが、営業時間外です。"
        logger.info("Reservation time processed: %s", reservation_time)
    else:
        fulfillment_text = "予約時間を指定してください。"
        logger.warning("No reservation time specified")

    return {"fulfillmentText": fulfillment_text}


def log_dict(log_dict):
    pprint(json.dumps(log_dict, ensure_ascii=False))
