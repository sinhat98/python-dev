.PHONY: run run-nemo run-default build build-nemo build-default

# サブターゲットを呼び出す条件をチェック
run: 
ifneq ($(filter nemo,$(MAKECMDGOALS)),)
	$(MAKE) run-nemo
else
	$(MAKE) run-default
endif

# デフォルトのdocker-compose runを実行するターゲット
run-default:
	docker compose run $(SERVICE_NAME)

# nemo用のdocker-composeファイルを使用して実行するターゲット
run-nemo:
	docker compose -f docker-compose-nemo.yaml run $(SERVICE_NAME)

# buildターゲットも同様のロジックを使用
build:
ifneq ($(filter nemo,$(MAKECMDGOALS)),)
	$(MAKE) build-nemo
else
	$(MAKE) build-default
endif

# デフォルトのdocker-compose buildを実行するターゲット
build-default:
	docker compose build $(SERVICE_NAME)

# nemo用のdocker-composeファイルを使用してビルドするターゲット
build-nemo:
	docker compose -f docker-compose-nemo.yaml build $(SERVICE_NAME)

# ダミーターゲット定義
nemo:
	@:  # 空のターゲットを定義して、'make run nemo'と'build nemo'を可能にする