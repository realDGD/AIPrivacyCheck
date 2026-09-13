#!/usr/bin/env python3
"""Deterministic Built-in Negative Corpus generator (v0.6.5 freeze gate).

Emits tests/fixtures/builtin_negative_corpus.jsonl:
  {"id","category","text","expect_detection": false}
  (+ the separate "format_perfect_fake" bucket: expect_detection=true BY
   DESIGN - offline deterministic detection cannot reject format-perfect
   fakes; reported separately, never counted into FPR.)

Everything is synthetic. Hard negatives follow the v0.6.5 freeze-gate spec
(code identifiers, vendor near-misses, config examples, adversarial
identifier-shaped runs).
"""

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "builtin_negative_corpus.jsonl"
SAMPLES = []  # (category, text)


def add(category, *texts):
    for t in texts:
        SAMPLES.append((category, t))


# --- 1. code identifiers / method / class / variable / package names -------
C = "code_identifiers"
add(C, "const secret = ModelUtils.toStringSafe(foo)", "obj.secret = getSecret()",
    "passwordManager 同步完成", "if (user.tokenizer) return", "token_count >= 64",
    "api_key_name VARCHAR(64)", "ACCESS_KEY_LENGTH = 40", "AUTH_TOKEN_TYPE bearer",
    "class UserSecretConfig(BaseModel)", "private_key_path = Path(...)".replace("Path(...)", "settings.KEY_PATH"),
    "SecretService.get_instance()", "SecretKeyFactory.getInstance()",
    "self._access_key_length = 40", "setPassword(value) 后请刷新会话",
    "getPassword() 返回 None", "ConfigurationManager.load(path)",
    "import os, sys, json, hashlib", "from cryptography.hazmat.primitives import hashes",
    "requests.post(url, headers=headers)", "logger.warning('cache miss: %s', key_name)",
    "def rotate_token_count(handler):", "if (result.is_valid) { emit(row); }",
    "SELECT secret_name FROM vault_metadata", "pip install python-dateutil",
    "npm i lodash-es", "org.apache.commons.lang3.StringUtils",
)

# --- 2. uuid / hash / sha / checksums --------------------------------------
C = "uuid_hash"
add(C, "id=550e8400-e29b-41d4-a716-446655440000", "commit 9c4d3f7a2b81e5d0614c7fa8b3e2d1094a5f6c7d",
    "sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "etag W/\"07f1b8c4-9d2e\"", "md5 5d41402abc4b2a76b9719d911017c592",
    "uuid4() 生成 8-4-4-4-12 格式", "checksum crc32=0x8a914b2c",
    "blob 51a8c4f2e9d3b7a60415c8e9f2d3b4a506172839", "trace_id=4bf92f3577b34da6a3ce929d0e0e4736",
    "digest sha512 9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043")

# --- 3. version / timestamp / dates ----------------------------------------
C = "version_timestamp"
add(C, "version 10.2.3.4 发布", "release v2.13.0-rc.1", "semver 1.0.0-alpha.1+build.7",
    "build 2026.09.13-08", "2026-09-13T09:15:00Z", "unix ts 1789312500",
    "updated_at 1789312500123", "日期 2026-12-01 08:30", "npm engine node >=20.11.0",
    "chrome/126.0.6478.61", "内核 6.8.0-45-generic", "镜像 build 88127341",
    "1700000000 与 1800000000 的时间戳比对", "commit dated 2026-09-12")

# --- 4. url / domain / hostname --------------------------------------------
C = "url_domain"
add(C, "文档 https://docs.example.com/guide/install", "镜像 registry.cn-hangzhou.example.com",
    "访问 https://example.org/docs/api#section-2", "git clone https://github.com/example/repo.git",
    "https://blog.example.org/2026/09/公共文章", "ws://localhost:8765/ws",
    "内网域名 nas.example.internal", "https://mirrors.aliyun.com/pypi/simple/",
    "访问 https://huggingface.co/Qwen/Qwen3-0.6B 页面", "站点 maps.example.cn/maps?q=杭州")

# --- 5. documentation / placeholder examples --------------------------------
C = "doc_examples"
add("reserved_documentation",
    "示例：test@example.com", "联系 admin@example.org",
    "EMAIL_USER=you@example.net",
    "占位格式 {{EMAIL_01_ab12cd34}}", "config.password = os.environ['DB_PASS']")
add(C, "api_key = <your-api-key>", "token: ${VAULT_SECRET_REF}", "password = ${DB_PASSWORD}",
    "Authorization: Bearer <token>", "secret: [FROM_ENV]", "password: ******（见保险箱）",
    "export OPENAI_API_KEY=sk-...", "api_key=\"REPLACE_ME\"",
    "使用 {{PLACEHOLDER}} 语法", "占位格式 {{EMAIL_01_ab12cd34}}",
    "config.password = os.environ['DB_PASS']")

# --- 6. fake credentials (weak near-misses, must NOT match) -----------------
C = "fake_credentials_weak"
add(C, "ghp_example", "github_pat_example", "hf_model_name", "sk-example", "sk-example-123",
    "AKIAEXAMPLE", "ASIAEXAMPLE123", "LTAIexample12", "AKIDexample", "xoxb-example",
    "xoxp-example", "xoxr-example-token", "glpat_example", "dapi_example",
    "lin_api_example", "8Q~example", "cli_example_secret", "sk-ant-example",
    "AIza EXAMPLE", "wJalrXUtnFEMI EXAMPLE KEY", "Bearer <credentials>",
    "密钥示例：***", "口令：******", "密码：[已隐藏]")

# --- 7. scientific / measurement data ---------------------------------------
C = "scientific_data"
add(C, "浓度 3.2e-5 mol/L", "质量 4.184 J/(g·K)", "测得 1.29e9 个粒子",
    "p < 0.0001，n=42", "pi=3.141592653589793", "光速 2.99792458e8 m/s",
    "误差 ±0.000000123", "x=1.23456789e12", "温度 298.15K", "电压 3.3V 电流 0.5A")

# --- 8. random / order / tracking ids ---------------------------------------
C = "random_ids"
add(C, "订单号 2026121312345678", "运单 SF138 0001 2233", "流水号 NO.20261213009812",
    "批次 BATCH-20261213-A1", "工单 2026121300123", "序列号 SN881234567890",
    "编号 AB1234567890", "参考号 REF99118822", "确认码 882199", "取件码 6-3-2211")

# --- 9. shell / environment -------------------------------------------------
C = "shell_env"
add(C, "export PATH=/usr/local/bin:$PATH", "echo $HOME", "kubectl get pods -n prod",
    "grep -r 'TODO' ./src", "sudo systemctl restart nginx", "ls -la | wc -l",
    "docker ps --format '{{.Names}}'", "cat /etc/os-release | head -2",
    "find . -name '*.py' -exec wc -l {} +", "ssh -L 8080:localhost:80 admin@bastion")

# --- 10. config examples (JSON/YAML/TOML/.env without real secrets) ---------
C = "config_examples"
add(C, '{"database": {"host": "db.prod", "port": 5432, "ssl": true}}',
    "server:\n  port: 8080\n  workers: 4",
    "[logging]\nlevel = \"info\"\nfile = \"/var/log/app.log\"",
    "DB_HOST=db.internal\nDB_PORT=5432\nDB_SSLMODE=verify-full",
    '{"model": "qwen3.5-0.8b", "temperature": 0, "max_tokens": 256}',
    "cache:\n  ttl: 300\n  max_entries: 10000",
    "region = \"cn-hangzhou\"\nprofile = \"default\"",
    '{"retries": 3, "backoff_ms": [100, 400, 1600]}',
    "timeout: 30s\nkeepalive: 60s")

# --- 11. sql / schema --------------------------------------------------------
C = "sql_schema"
add(C, "CREATE TABLE users (id BIGINT PRIMARY KEY, created_at TIMESTAMPTZ);",
    "SELECT count(*) FROM orders WHERE status = 'paid';",
    "ALTER TABLE sessions ADD COLUMN expires_at TIMESTAMPTZ;",
    "INSERT INTO audit(event, ts) VALUES ('login', now());",
    "CREATE INDEX idx_orders_user ON orders(user_id);",
    "-- 迁移脚本：调整字符集 utf8mb4", "GRANT SELECT ON reports TO analyst;")

# --- 12. model / api / library names ----------------------------------------
C = "model_api_names"
add(C, "模型 gpt-4o 与 claude-3-5-sonnet 对比", "qwen3.5-0.8b 本地可跑",
    "调用 /v1/chat/completions 接口", "使用 transformers 4.57.6 加载",
    "库版本 flask==3.1.0 sqlalchemy==2.0.36", "模型路径 /models/qwen3.5-2b",
    "qwen/qwen3.5-2b 是 text-generation 类型", "API 端点 /api/v2/detect",
    "module tensorflow.keras.layers", "torch.cuda.is_available()")

# --- 13. network-like / number-like adversarial -----------------------------
C = "network_like"
add(C, "非法地址 999.999.999.999", "五段伪 IP 1.2.3.4.5", "CIDR 写法 10.0.0.0/8",
    "网段 172.16.0.0/12 说明", "semver 10.20.30",
    "电话样例 10086", "号码样例 12345678901", "卡样例 0000000000000000",
    "证件样例 000000000000000", "时间戳样例 1380013800000", "长度 16 的数字 1234567890123456",
    "UUID 样例 123e4567-e89b-12d3-a456-426614174000", "哈希样例 a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
    "ipv6 示例 fe80::1%eth0", "网站 192.0.2.1 是文档保留段")
add("format_perfect_fake",
    "1:2:3:4:5:6:7:8",  # structurally a valid IPv6 - indistinguishable
    "12:34:56:78:9a:bc")  # locally-administered MAC - indistinguishable

# --- 14. CJK context sentences (credential words, no values) ----------------
C = "cjk_context"
add(C, "密码策略要求至少 16 个字符。", "请勿在聊天中泄露凭据。",
    "本次演练不涉及真实凭证。", "访问密钥请放入保险箱。",
    "口令到期前 7 天会提醒。", "私钥文件已用 TPM 保护。",
    "用户名是登录身份的一部分，不是秘密。", "账号被锁定 30 分钟。",
    "令牌轮换策略每月执行。", "秘钥管理系统上线了。")


# --- 扩充批次（保持多样性，禁止同构复制）------------------------------------
add("code_identifiers",
    "hashed = hashlib.sha256(payload).hexdigest()", "self.token_bucket.refill()",
    "await verifier.verify(jwt_claims)", "config.get('timeout', 30)",
    "class SecretRotator:", "async def fetch_quota(self):",
    "user.settings.password_policy.enabled", "Route::post('/rotate', [TokenController::class, 'rotate'])",
    'fmt.Sprintf("%s=%v", keyName, keyLength)', "try { validator.validate(input); }",
    "for k, v in sorted(envelope.items()):", "printf('%s: %d\n', name, count)",
    "public static final int API_KEY_LENGTH = 64;", "func (s *Store) Put(ctx context.Context)",
    'keyId := os.Getenv("KEY_ID")', "assert response.status_code == 200",
    "JSON.stringify({event: 'rotate', ok: true})", "DELETE FROM tokens WHERE revoked = true",
    "set -euo pipefail", "ln -sf target link_name",
    "hasattr(module, 'SECRET_NAME')", "match (kind) { case 'id': break; }",
    "console.log('token refresh scheduled')", "tokens.map(t => t.prefix)",
    "with vault.client() as c:", "@dataclass(frozen=True)")
add("uuid_hash",
    "hash 00000000000000000000000000000000", "run_id 7e5a1c90-4b2f-4c3d-8d0a-9f1e2d3c4b5a",
    "digest blake2b a8cb2d4f0e13b6598701c2fa4d5e6b708192a3b4c5d6e7f8091a2b3c4d5e6f70",
    "etag 33a64df5514ffccf832fc45d4a3b7c", "job 20261213-abcdef-4210",
    "session fb350c1e-8a11-42bb-9c63-6b3f1e0d7a22", "request id req_01HQX7P4K2M3N4",
    "sha1 85d7e5f1c2b3a49687061524396b7a8c9d0e1f2a", "签名摘要 4f2a6b8c0d1e3f5a7b9c1d3e5f7a9b1d",
    "object id 2a9f8b7c6d5e4f3a2b1c0d9e8f7a6b5c", "批次指纹 f0e1d2c3b4a5968778695a4b3c2d1e0f")
add("version_timestamp",
    "alpha build 0.0.1", "beta 1.2.0-beta.3", "lts 22.04.5", "patch 3.0.186",
    "ts 1789312500789", "modified 2026-12-31T23:59:59+08:00",
    "发布于 2027 年 1 月 4 日", "effective 2027-02-01", "start=2026-01-01, end=2026-12-31",
    "epoch 1900000000", "version: 8.25.0", "kernel 15.2-3", "fw 2.1.37",
    "发行 2026Q4", "定时 00:00 UTC", "backup at 03:00, 15:00")
add("url_domain",
    "https://api.example.com/v2/health", "ldap://directory.example.org/dc=corp",
    "s3://artifacts-bucket/releases/", "file:///etc/example/config.yaml",
    "https://gitlab.example.cn/-/settings/repository",
    "https://pypi.org/project/requests/", "mirror.aliyuncs.com/docker-library",
    "https://modelscope.cn/models/Qwen/Qwen3-0.6B", "host backup.example.net:2222",
    "ws://127.0.0.1:9000/stream", "https://example.com/zh/guide#安装",
    "https://charts.example.io/stable/index.yaml", "ftp://files.example.com/pub/README",
    "proxy http://squid.internal:3128",
    "保留段示例 (见 reserved_documentation 桶)")
add("doc_examples",
    'connect "host=<db-host> port=<db-port>"', "secretRef: vault://kv/data/example",
    "api-key: ${secrets.API_KEY}", "TOKEN_FILE=/run/secrets/token",
    "use your-tenant-id here", "password: (from prompt)", "accessKey: <ACCESS_KEY>",
    "示例 2：sk-***-***", "写法：Bearer <jwt>", "placeholder: <<REDACTED>>",
    'config: {"log": "info", "key": null}', "示例输出：OK (200)",
    "替换 <your-domain> 为实际域名")
add("fake_credentials_weak",
    "gho_example", "ghu_example", "ghr_example", "hf_token_example_long",
    "sk-proj-EXAMPLE", "sk-ant-example", "ASIAEXAMPLEKEY00", "ABIAEXAMPLEKEY00",
    "xoxs-example", "xoxa-example", "xapp-example", "xwfp-example",
    "glptt-example", "hooks.slack.com 例句", "dapiEXAMPLE", "lin_api_EXAMPLE",
    "cliEXAMPLE", "AIzaEXAMPLE", "rk_live_EXAMPLE", "sk_live_EXAMPLE",
    "github_pat_EXAMPLE", "密钥=example", "令牌：EXAMPLE", "凭据 example 文案")
add("scientific_data",
    "半径 6.371e6 m", "频率 2.45 GHz", "浓度 1.0e-3 %", "误差 1e-11 s",
    "计数 1234567890 ± 42", "增益 0.9998", "斜率 -3.7e-4", "能量 6.626e-34 J·s",
    "采样 48000 Hz", "码率 1.5e7 bps", "质量比 1:3.6e2", " pH 7.40",
    "坐标 3.14e15 m", "带宽 1e9 Hz", "分辨 1.22e-7 rad")
add("random_ids",
    "编号 ZX-99-887766", "单号 YT4213887766", "回执 RC-8877-6655",
    "操作流水 OP202612130001", "会话序号 0042", "分片 17/64",
    "批次号 LOT-88-12-34", "卡位 A-12-34", "库位 K-09-88-2",
    "模板 TPL-1001", "规则 R-77", "案例 CS-2026-0042")
add("shell_env",
    "sed -i 's/old/new/g' file", "awk '{print $2}' data.txt",
    "tar -czf backup.tgz /data", "chown app:app /run/app", "mount | grep /data",
    "journalctl -u app -f", "ip -brief addr show", "ss -tlnp | head",
    "curl -s https://api.example.com/health", "htop -u app",
    'setx PATH "%PATH%;C:\\tools"', "Get-ChildItem Env: | head",
    'echo "export EDITOR=vim" >> ~/.bashrc', "chmod +x deploy.sh",
    "pushd /var/log && ls", "xargs -n1 echo < list.txt")
add("config_examples",
    '{"enable_tls": true, "min_version": "1.3"}',
    "workers: 8\nmax_connections: 500",
    '[server]\nbind = "0.0.0.0:8443"\nbacklog = 1024',
    "LOG_LEVEL=debug\nMETRICS_PORT=9100",
    '{"queue": {"type": "memory", "limit": 1000}}',
    "replicas: 3\nstrategy: rolling",
    'interval = "5m"\njitter = "30s"',
    "db:\n  pool: 10\n  idle: 60",
    '{"name": "edge", "weight": 0.5}',
    'log_format = "json"\ntz = "Asia/Shanghai"',
    "retry:\n  max: 5\n  policy: exponential")
add("sql_schema",
    "SELECT id, name FROM products WHERE price > 100 ORDER BY id LIMIT 10;",
    "CREATE VIEW v_active AS SELECT * FROM users WHERE active;",
    "TRUNCATE TABLE staging_metrics;",
    "CREATE TYPE order_status AS ENUM ('open','closed');",
    "EXPLAIN ANALYZE SELECT 1;", "VACUUM ANALYZE orders;",
    "CREATE TRIGGER trg_touch BEFORE UPDATE ON sessions FOR EACH ROW EXECUTE fn_touch();",
    "COMMENT ON COLUMN users.phone IS 'masked at write time';",
    "SELECT pg_size_pretty(pg_total_relation_size('events'));",
    "BEGIN; UPDATE quotas SET used = used + 1; COMMIT;")
add("model_api_names",
    "接口 GET /healthz 返回 200", "库 pandas 2.2.3", "框架 django==5.1.1",
    "qwen2.5-0.5b-instruct 量化版", "模型卡 modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct",
    "embedding 模型 bge-small-zh-v1.5", "运行 uvicorn app:main --port 9000",
    "包 langchain-core==0.3.7", "端点 POST /v1/messages",
    "服务 NATS topics：events.detect", "模块包名 privacy.detectors",
    "类名 SpanResolver", "函数 detect_entities(text)",
    "OpenAI 兼容端点 /v1/completions 说明", "CLI 用法 vault-engine scrub INFILE")
add("network_like",
    " malformed ::::::", "版本样例 192.168.1",
    "三段数字 10.20.30", "四段 999.1.2.3", "掩码写法 /24",
    "MAC 样例 00-00-00-00-00-00", "MAC 样例 ff:ff:ff:ff:ff:ff",
    "纯数字 0000000000", "学号样例 2026000000", "票号 9999999999",
    "hex 0000ffff", "binary 0b10101010", "octal 0o777",
    "UTF-8 码点 U+1F600", "区间 [0,1)", "百分比 99.99%")
add("cjk_context",
    "秘钥轮换窗口为每周二 02:00。", "凭证上传通道已关闭。",
    "授权码单次有效期为 10 分钟。", "访问密钥分两级：读写与只读。",
    "接口密钥需绑定白名单。", "私钥绝不应出现在日志中。",
    "口令复杂度不做强制要求。", "该账号已启用双因子认证。",
    "令牌过期后需重新登录。", "凭证夹带将被网关拦截。",
    "密码找回流程见帮助中心。", "签名密钥与加密密钥分离存放。")

# --- 15. format-perfect fakes (SEPARATE bucket; expected detection) ---------
import random
random.seed(20260913)
def rchars(alphabet, n):
    return "".join(random.choice(alphabet) for _ in range(n))
UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
AL = "abcdefghijklmnopqrstuvwxyz0123456789"
HEX = "0123456789abcdef"
add("format_perfect_fake",
    "ghp_" + rchars(AL + "_", 36),
    "github_pat_" + rchars(AL, 22) + "_" + rchars(AL, 59),
    "hf_" + rchars(AL, 34),
    "sk-" + rchars(AL, 48),
    "AKIA" + rchars(UP, 16),
    "LTAI" + rchars(AL, 16),
    "glpat-" + rchars(AL + "-_", 20),
    "cli_" + rchars("abcdefghijklmnopqrstuvwxyz0123456789", 16),
    "ab1" + "8Q~" + rchars("aZ09_~-." + "", 34),
    "密钥：" + rchars("ABCDEFGH12345678", 12),
    "password=" + rchars("abcdefgh0123456789", 16),
    "Bearer " + rchars(AL, 40),
)


def main():
    out = []
    counters = {}
    for cat, text in SAMPLES:
        counters[cat] = counters.get(cat, 0) + 1
        out.append({
            "id": f"neg_{len(out)+1:03d}",
            "category": cat,
            "text": text,
            "expect_detection": cat in ("format_perfect_fake", "reserved_documentation"),
        })
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"corpus written: {FIXTURE} ({len(out)} samples)")
    for cat, n in counters.items():
        print(f"  {cat:<22} {n}")
    clear = sum(n for c, n in counters.items() if c != "format_perfect_fake")
    print(f"clearly-negative: {clear} | format-perfect fakes: {counters.get('format_perfect_fake', 0)}")


if __name__ == "__main__":
    main()
