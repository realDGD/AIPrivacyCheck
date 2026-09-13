#!/usr/bin/env python3
"""Deterministic generator for the 100-document Privacy Benchmark v2 corpus.

Freezes tests/fixtures/privacy_benchmark_v2_100.jsonl. The fixture is the
frozen artifact: benchmarks must NOT regenerate it at runtime. Re-running
this script with the same fixed seed reproduces the same corpus bit-for-bit.

Corpus contract (v0.6.5 goal):
- 100 fully synthetic documents (no real personal data, no real secrets).
- Fixed composition: 30 zh short/medium, 20 zh long, 20 en short/medium,
  10 en long, 10 mixed/structured, 10 PII-free/public/ambiguous.
- Detection gold (entities) is separate from redaction policy
  (per-entity should_redact + context_class) and from the semantic
  privacy layer (semantic_privacy entries with a sensitivity verdict).
- Every entity span satisfies text[start:end] == text fragment.
"""

from pathlib import Path
import json

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "privacy_benchmark_v2_100.jsonl"

DOCS = []


def D(case_id, language, length_class, domain, text, ents, semantic=None,
      risk_group=None, notes=""):
    """Registers one document; computes spans for all fragments.

    ents: list of (type, fragment, should_redact, context_class).
    semantic: list of (type, fragment, sensitive, context_class).
    Fragments capture ALL occurrences (repeated identities are repeated gold).
    """
    entities = []
    for etype, frag, redact, cls in ents:
        idx = 0
        hits = 0
        while True:
            pos = text.find(frag, idx)
            if pos == -1:
                break
            entities.append({
                "text": frag,
                "start": pos,
                "end": pos + len(frag),
                "type": etype,
                "should_redact": redact,
                "context_class": cls,
            })
            idx = pos + len(frag)
            hits += 1
        assert hits > 0, f"{case_id}: fragment {frag!r} not found in text"
    semantic_privacy = []
    for stype, frag, sensitive, cls in (semantic or []):
        idx = 0
        hits = 0
        while True:
            pos = text.find(frag, idx)
            if pos == -1:
                break
            semantic_privacy.append({
                "text": frag,
                "start": pos,
                "end": pos + len(frag),
                "type": stype,
                "sensitive": sensitive,
                "context_class": cls,
            })
            idx = pos + len(frag)
            hits += 1
        assert hits > 0, f"{case_id}: semantic fragment {frag!r} not found"
    pii_free = not entities
    DOCS.append({
        "id": case_id,
        "language": language,
        "length_class": length_class,
        "domain": domain,
        "text": text,
        "entities": entities,
        "semantic_privacy": semantic_privacy,
        "pii_free": pii_free,
        "risk_group": risk_group or ("quasi_identifier" if any(c == "quasi_identifier" for _, _, _, c in ents) else ("direct" if entities or semantic_privacy else "none")),
        "notes": notes,
    })



def _id_card_checksum(first17: str) -> str:
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checksum = "10X98765432"
    total = sum(int(c) * w for c, w in zip(first17, weights))
    return checksum[total % 11]


def _make_id_card(first17: str) -> str:
    return first17 + _id_card_checksum(first17)

def P(*paragraphs):
    return "\n".join(paragraphs)

def EXTEND(case_id, extra_text):
    """Appends extension text to a registered doc and re-scans gold fragments.

    Fragments re-occurring in the extension gain additional gold spans with
    correct offsets (realistic repeated mentions).
    """
    doc = next(d for d in DOCS if d["id"] == case_id)
    base_len = len(doc["text"])
    doc["text"] = doc["text"] + extra_text

    # Deduplicate by fragment text: the entity list holds one entry per
    # occurrence, and scanning the appendix once per occurrence would append
    # the same new span once per duplicate entry.
    entity_specs = {}
    for ent in doc["entities"]:
        entity_specs.setdefault(ent["text"], ent)
    for frag, sample in entity_specs.items():
        idx = 0
        while True:
            pos = extra_text.find(frag, idx)
            if pos == -1:
                break
            doc["entities"].append({
                "text": frag,
                "start": base_len + pos,
                "end": base_len + pos + len(frag),
                "type": sample["type"],
                "should_redact": sample["should_redact"],
                "context_class": sample["context_class"],
            })
            idx = pos + len(frag)

    semantic_specs = {}
    for sem in doc["semantic_privacy"]:
        semantic_specs.setdefault(sem["text"], sem)
    for frag, sample in semantic_specs.items():
        idx = 0
        while True:
            pos = extra_text.find(frag, idx)
            if pos == -1:
                break
            doc["semantic_privacy"].append({
                "text": frag,
                "start": base_len + pos,
                "end": base_len + pos + len(frag),
                "type": sample["type"],
                "sensitive": sample["sensitive"],
                "context_class": sample["context_class"],
            })
            idx = pos + len(frag)


EXTENSIONS = {}


def _ext(case_id):
    def _register(paragraphs):
        EXTENSIONS.setdefault(case_id, []).append(P(*paragraphs))
    return _register



# ---------------------------------------------------------------------------
# case_001-030: Chinese short / medium
# ---------------------------------------------------------------------------

# -- The four hard PERSON/USERNAME cases (goal 1.3, verbatim golds) --
D("case_001", "zh", "short", "conversation",
  "张三今天来了，稍后张三又打电话过来。",
  [("CN_NAME", "张三", True, "private_person")],
  notes="HARD CASE: narrative name mentions must be CN_NAME, never USERNAME")

D("case_002", "zh", "short", "conversation",
  "张三，用户名是李四。",
  [("CN_NAME", "张三", True, "private_person"),
   ("USERNAME", "李四", True, "account_handle")],
  notes="HARD CASE: 张三=CN_NAME, 李四=USERNAME")

D("case_003", "zh", "short", "account_note",
  "用户名是张三。",
  [("USERNAME", "张三", True, "account_handle")],
  notes="HARD CASE: labeled username value")

D("case_004", "zh", "short", "account_note",
  "张三是用户名。",
  [("USERNAME", "张三", True, "account_handle")],
  notes="HARD CASE: identified-as-username")

# -- Detection vs redaction pairs (goal 1.4, verbatim golds) --
D("case_005", "zh", "short", "customer_service",
  "客服电话是10086。",
  [("PHONE", "10086", False, "public_hotline")],
  notes="detected PHONE, must NOT be redacted")

D("case_006", "zh", "short", "note",
  "我的手机号是13800000000。✅",
  [("CN_PHONE_NUMBER", "13800000000", True, "private_contact")],
  notes="private mobile: detect AND redact")

D("case_007", "zh", "short", "it_note",
  "公共 DNS 是 8.8.8.8。",
  [("IP_ADDRESS", "8.8.8.8", False, "public_infrastructure")],
  notes="detected IP, must NOT be redacted")

D("case_008", "zh", "short", "it_note",
  "我的 NAS 内网地址是 192.168.1.50。",
  [("IP_ADDRESS", "192.168.1.50", True, "private_network")],
  notes="private NAS IP: detect AND redact")

D("case_009", "zh", "short", "tutorial",
  "示例邮箱为 test@example.com。",
  [("EMAIL", "test@example.com", False, "documentation_example")],
  notes="RFC2606 example: detect, do not redact")

D("case_010", "zh", "short", "email_signature",
  "我的联系邮箱为 xiaolin@example.com。",
  [("EMAIL", "xiaolin@example.com", True, "private_contact")],
  notes="personal mailbox: detect AND redact")

D("case_011", "zh", "short", "delivery",
  "收件人陈晓，电话13755556666，地址杭州市西湖区文二路39号。｜包裹🈶易碎品",
  [("CN_NAME", "陈晓", True, "private_person"),
   ("CN_PHONE_NUMBER", "13755556666", True, "private_contact"),
   ("CN_ADDRESS", "杭州市西湖区文二路39号", True, "private_address")])

D("case_012", "zh", "short", "bank_sms",
  "您尾号3366的储蓄卡12月28日20:05消费500.00元(余额20000.00元)。",
  [],
  notes="masked card tail only: no detection gold; counts as ambiguous public doc")

D("case_013", "zh", "short", "chat",
  "王芳😀明天下午三点来公司做入职体检，记得带身份证原件。",
  [("CN_NAME", "王芳", True, "private_person")],
  notes="ID card mentioned without a number: no numeric gold entity")

D("case_014", "zh", "short", "hotel",
  "酒店入住人：赵敏，预订确认号 HTL88213。",
  [("CN_NAME", "赵敏", True, "private_person"),
   ("RECORD_ID", "HTL88213", True, "booking_reference")])

D("case_015", "zh", "short", "school",
  "学生周雨桐的学号是20260118。",
  [("CN_NAME", "周雨桐", True, "private_person"),
   ("STUDENT_ID", "20260118", True, "school_record")])

D("case_016", "zh", "medium", "meeting_minutes",
  P("项目周会纪要（节选）",
    "时间：12月19日 10:00-11:30；主持：产品负责人林岚。",
    "参会：研发吴凯、测试郑爽、客服主管刘畅。",
    "会后请把客户投诉工单号 TS-20261219-003 发给值班邮箱 duty@example-corp.cn。",
    "下周现场支持安排在滨江区物联网街 528 号客户机房，联系人高翔 13900138999。"),
  [("CN_NAME", "林岚", True, "private_person"),
   ("CN_NAME", "吴凯", True, "private_person"),
   ("CN_NAME", "郑爽", True, "private_person"),
   ("CN_NAME", "刘畅", True, "private_person"),
   ("RECORD_ID", "TS-20261219-003", True, "ticket_reference"),
   ("EMAIL", "duty@example-corp.cn", False, "corporate_role_mailbox"),
   ("CN_ADDRESS", "滨江区物联网街 528 号客户机房", True, "business_location"),
   ("CN_NAME", "高翔", True, "private_person"),
   ("CN_PHONE_NUMBER", "13900138999", True, "private_contact")])

D("case_017", "zh", "medium", "support_ticket",
  P("工单 #AFT-88123 处理记录",
    "客户孙女士（回拨电话 0571-88112266）反馈路由器频繁掉线。",
    "工程师初步判断为固件 bug，建议远程升级。",
    "远程协助地址 https://support.example-corp.cn/session/88123，仅限工单有效期使用。",
    "客户家庭宽带账号 hz_sun_2024，运营商登记人孙一鸣。"),
  [("RECORD_ID", "#AFT-88123", True, "ticket_reference"),
   ("CN_LANDLINE", "0571-88112266", True, "private_contact"),
   ("PRIVATE_URL", "https://support.example-corp.cn/session/88123", True, "session_url"),
   ("USERNAME", "hz_sun_2024", True, "account_handle"),
   ("CN_NAME", "孙一鸣", True, "private_person")])

D("case_018", "zh", "medium", "resume",
  P("求职者简历（脱敏前存档片段）",
    "姓名：郑洁；性别：女；出生日期：1996年4月12日。",
    "现居：成都市高新区天府三街 199 号 A 座 12 层（Cafe\u0301 楼层标牌复检 naif\u0308ve 拼写已核对）。",
    "联系方式：zhengjie.1996@example-mail.com / 15900012233。",
    "工作经历：2019-2023 任某互联网公司后端工程师；2023 年至今任某银行科技子公司高级工程师。"),
  [("CN_NAME", "郑洁", True, "private_person"),
   ("CN_BIRTH_DATE", "1996年4月12日", True, "birth_date"),
   ("CN_ADDRESS", "成都市高新区天府三街 199 号 A 座 12 层", True, "private_address"),
   ("EMAIL", "zhengjie.1996@example-mail.com", True, "private_contact"),
   ("CN_PHONE_NUMBER", "15900012233", True, "private_contact")])

D("case_019", "zh", "medium", "loan_application",
  P("个人消费贷申请材料清单（客户联）",
    "申请人：冯磊，身份证号 33010519880214551X。",
    "工作单位：杭州市某物流公司；月收入 18,000 元；名下负债：车贷余额 6.2 万元。",
    "本人确认以上信息真实，并授权贵行查询征信。",
    "若有疑问请致电客户经理 13766668888。"),
  [("CN_NAME", "冯磊", True, "private_person"),
   ("CN_ID_CARD", "33010519880214551X", True, "government_identity"),
   ("FINANCIAL", "车贷余额 6.2 万元", True, "debt_info")],
  [("FINANCIAL", "个人消费贷申请", True, "loan_context"),
   ("FINANCIAL", "月收入 18,000 元", True, "income_info")])

D("case_020", "zh", "medium", "property",
  P("二手房带看记录",
    "房源：上城区江城路 88 号 3 幢 2 单元 501；挂牌价 468 万。",
    "看房客户：何俊杰（配偶同行），联系电话 13588889999。",
    "中介备注：客户置换需求，现房在拱墅区大关路 25 号，贷款已结清。"),
  [("CN_ADDRESS", "上城区江城路 88 号 3 幢 2 单元 501", True, "property_address"),
   ("CN_NAME", "何俊杰", True, "private_person"),
   ("CN_PHONE_NUMBER", "13588889999", True, "private_contact"),
   ("CN_ADDRESS", "拱墅区大关路 25 号", True, "private_address")])

D("case_021", "zh", "medium", "medical_outpatient",
  P("门诊病历（内科）",
    "患者许涛，男，41 岁。",
    "主诉：近两周多饮多尿，体重下降 5 公斤。",
    "诊断：2 型糖尿病；处置：二甲双胍 0.5g 每日两次，两周后复查空腹血糖。",
    "家属联系电话 13811112222。"),
  [("CN_NAME", "许涛", True, "patient_identity")],
  [("MEDICAL", "2 型糖尿病", True, "diagnosis"),
   ("MEDICAL", "空腹血糖", True, "lab_indicator")])

D("case_022", "zh", "medium", "insurance",
  P("车险理赔回访记录",
    "被保险人：唐薇；保单号 PC-2026-88213。",
    "出险时间：12 月 10 日 18:40；地点：余杭区文一西路 969 号地下车库。",
    "车辆：浙A·D88213，左后门剐蹭。",
    "定损金额 2,300 元，赔款将支付至尾号 8821 的银行卡。"),
  [("CN_NAME", "唐薇", True, "private_person"),
   ("RECORD_ID", "PC-2026-88213", True, "policy_reference"),
   ("CN_LICENSE_PLATE", "浙A·D88213", True, "vehicle_identifier")])

D("case_023", "zh", "medium", "travel",
  P("出差行程报备",
    "出差人：宋佳明、钱多余；目的地：西安市雁塔区丈八一路 1 号客户总部。",
    "航班 MU2216，12 月 22 日 08:35 起飞。",
    "住宿：锦业路如家酒店，预订人手机 18966667777。",
    "预算合计 12,800 元，含往返机票与住宿。"),
  [("CN_NAME", "宋佳明", True, "private_person"),
   ("CN_NAME", "钱多余", True, "private_person"),
   ("CN_ADDRESS", "西安市雁塔区丈八一路 1 号客户总部", True, "business_location"),
   ("CN_PHONE_NUMBER", "18966667777", True, "private_contact")])

D("case_024", "zh", "medium", "community_notice",
  P("物业通知（3 号楼）",
    "尊敬的业主：本小区将于 12 月 25 日 09:00-17:00 进行消防管道检修。",
    "施工期间请将车辆挪至地面访客车位。",
    "物业值班电话 0571-87654321；紧急联系人：物业经理马国强。",
    "如有特殊情况请联系 3 号楼楼长秦阿姨（微信 hz_qin_1957）。"),
  [("CN_LANDLINE", "0571-87654321", False, "business_hotline"),
   ("CN_NAME", "马国强", True, "private_person"),
   ("CN_SOCIAL_ACCOUNT", "hz_qin_1957", True, "social_handle")])

D("case_025", "zh", "medium", "police_record_fictional",
  P("（模拟演练材料，非真实笔录）",
    "报警人：吕晓东；联系电话 13700001111。",
    "事发地点：西湖区文三路 100 号门口。",
    "陈述：12 月 18 日 21 时许，其电动车被盗，车辆牌号杭州 C·33881。",
    "损失预估 3,000 元；随车工牌一张（编号 EMP-10238）。"),
  [("CN_NAME", "吕晓东", True, "private_person"),
   ("CN_PHONE_NUMBER", "13700001111", True, "private_contact"),
   ("CN_LICENSE_PLATE", "杭州 C·33881", True, "vehicle_identifier"),
   ("EMPLOYEE_ID", "EMP-10238", True, "employee_record")])

D("case_026", "zh", "medium", "email",
  P("发件人：billing@shop.example.com",
    "收件人：yuan_yuan_88@personal-mail.cn",
    "主题：订单 88213999 支付提醒",
    "袁媛您好，您 12 月 20 日下单的商品尚未支付，订单金额 399 元。",
    "如已支付请忽略本邮件；客服电话 400-820-8820。"),
  [("EMAIL", "yuan_yuan_88@personal-mail.cn", True, "private_contact"),
   ("CN_NAME", "袁媛", True, "private_person"),
   ("RECORD_ID", "88213999", True, "order_reference"),
   ("PHONE", "400-820-8820", False, "business_hotline")])

D("case_027", "zh", "medium", "workflow",
  P("采购审批流（OA 系统导出）",
    "申请人：郭建波（采购部）；审批人：财务总监 蒋丽。",
    "采购内容：服务器 3 台，预算 96,000 元，供应商：杭州某云计算有限公司。",
    "供应商联系人：sales@cloud-hz.example.com，电话 0571-99887766。",
    "备注：合同编号 PO-2026-0301，签约后 30 个工作日内交付。"),
  [("CN_NAME", "郭建波", True, "private_person"),
   ("CN_NAME", "蒋丽", True, "private_person"),
   ("EMAIL", "sales@cloud-hz.example.com", False, "corporate_role_mailbox"),
   ("CN_LANDLINE", "0571-99887766", False, "business_hotline"),
   ("RECORD_ID", "PO-2026-0301", True, "contract_reference")])

D("case_028", "zh", "medium", "group_announcement",
  P("业委会筹备群公告",
    "各位业主：筹备组已完成首批志愿者登记。",
    "志愿者名单：韩梅（1 号楼）、曹阳（2 号楼）、沈飞（5 号楼）。",
    "材料递交地址：余杭区高兴路 66 号社区服务中心 203 室。",
    "咨询请加筹备微信 ye wei hui 2026 或联系 13755550001。"),
  [("CN_NAME", "韩梅", True, "private_person"),
   ("CN_NAME", "曹阳", True, "private_person"),
   ("CN_NAME", "沈飞", True, "private_person"),
   ("CN_ADDRESS", "余杭区高兴路 66 号社区服务中心 203 室", True, "business_location"),
   ("CN_PHONE_NUMBER", "13755550001", True, "private_contact")])

D("case_029", "zh", "medium", "cvs_log",
  P("系统变更记录",
    "变更单 CHG-2026-0912；执行人：运维值班 潘岩。",
    "涉及主机：db-primary（内网地址 10.8.0.15）。".replace("10.8.0.15", "10.8.0.15"),
    "变更窗口：12 月 24 日 01:00-03:00；回滚负责人：值班手机 15811112222。",
    "审批：基础设施部经理 邱敏。"),
  [("RECORD_ID", "CHG-2026-0912", True, "change_reference"),
   ("CN_NAME", "潘岩", True, "private_person"),
   ("IP_ADDRESS", "10.8.0.15", True, "internal_infrastructure"),
   ("CN_PHONE_NUMBER", "15811112222", True, "private_contact"),
   ("CN_NAME", "邱敏", True, "private_person")])

D("case_030", "zh", "medium", "quasi_researcher",
  P("学术会议通讯录条目（示例）",
    "姓名：丁一鸣；年龄 32 岁；职称：副教授。",
    "单位：北京某大学计算机学院；研究方向：联邦学习隐私保护。",
    "办公地址：北京市海淀区清华园路 27 号；实验室：智能计算楼 4 层 402。",
    "代表论文：Differentially Private Federated Analytics（2024）。",
    "阿拉伯语转写备案：دينغ يي مينغ；希腊字母编号 Α-8821。"),
  [("CN_NAME", "丁一鸣", True, "private_person")],
  risk_group="quasi_identifier",
  notes="age+position+institution+research+address combination re-identification risk",
  semantic=[])


# ---------------------------------------------------------------------------
# case_031-050: Chinese long (8 long + 2 very_long)
# ---------------------------------------------------------------------------

D("case_031", "zh", "long", "medical_record",
  P("住院记录（心内科，示例病历）",
    "患者蓝成蹊，女，58 岁，已婚，退休教师，因反复胸闷气促 3 个月，加重 1 周入院。",
    "既往史：高血压病史 10 年，长期服用苯磺酸氨氯地平；2 型糖尿病史 6 年，目前使用胰岛素控制血糖。",
    "入院查体：血压 156/94mmHg，心率 88 次/分，双下肢轻度水肿。",
    "辅助检查：心电图提示窦性心律，ST-T 段改变；心脏彩超示左室射血分数 48%。",
    "初步诊断：1. 冠心病 心功能 II 级；2. 高血压 3 级（很高危）；3. 2 型糖尿病。",
    "诊疗计划：完善冠脉造影评估，必要时行 PCI 术；继续降压、降糖、抗血小板治疗。",
    "主管医师：心内科主治医师 顾清源；责任护士 白鹭。",
    "家属联系方式：患者之子 蓝霆，电话 13622223333。",
    "入院日期：2026 年 12 月 5 日；记录医师签名：顾清源。"),
  [("CN_NAME", "蓝成蹊", True, "patient_identity"),
   ("CN_NAME", "顾清源", True, "private_person"),
   ("CN_NAME", "白鹭", True, "private_person"),
   ("CN_NAME", "蓝霆", True, "private_person"),
   ("CN_PHONE_NUMBER", "13622223333", True, "private_contact")],
  [("MEDICAL", "2 型糖尿病", True, "diagnosis"),
   ("MEDICAL", "冠心病 心功能 II 级", True, "diagnosis"),
   ("MEDICAL", "高血压 3 级", True, "diagnosis"),
   ("MEDICAL", "胰岛素控制血糖", True, "treatment"),
   ("MEDICAL", "冠脉造影", True, "procedure")])

D("case_032", "zh", "long", "legal",
  P("民事委托代理合同（示例文本，非真实案件）",
    "甲方（委托人）：姜文昊，身份证号 320102199203114432，住址南京市玄武区中山东路 300 号 02 幢 1801 室。",
    "乙方（受托人）：江苏某律师事务所。",
    "鉴于甲方与南京某置业有限公司之间存在商品房买卖合同纠纷，甲方委托乙方律师代理本案一审诉讼程序。",
    "第一条 委托事项：乙方指派律师倪思远作为甲方一审诉讼代理人，代理权限为特别授权。",
    "第二条 代理费用：本案代理费人民币 45,000 元，于本合同签订之日起五日内支付至乙方账户。",
    "第三条 甲方义务：甲方应当如实陈述案件事实，提供购房合同、付款凭证、微信沟通记录等证据材料。",
    "第四条 保密条款：乙方对甲方因本合同而知悉的个人隐私、商业秘密负有保密义务。",
    "第五条 联系方式：甲方联系电话 13951002233；乙方办案律师助理邮箱 lawassistant@jslaw.example.com。",
    "甲方签字：姜文昊；乙方盖章：江苏某律师事务所；签订日期：2026 年 12 月 1 日。"),
  [("CN_NAME", "姜文昊", True, "private_person"),
   ("CN_ID_CARD", "320102199203114432", True, "government_identity"),
   ("CN_ADDRESS", "南京市玄武区中山东路 300 号 02 幢 1801 室", True, "private_address"),
   ("CN_NAME", "倪思远", True, "private_person"),
   ("CN_PHONE_NUMBER", "13951002233", True, "private_contact"),
   ("EMAIL", "lawassistant@jslaw.example.com", False, "corporate_role_mailbox")],
  [("FINANCIAL", "代理费人民币 45,000 元", True, "contract_amount")])

D("case_033", "zh", "long", "hr",
  P("员工背景调查报告（模拟）",
    "候选人：乐嘉树，男，34 岁。",
    "一、教育背景：2010-2014 就读于武汉某大学软件工程专业，获学士学位。",
    "二、工作履历核查：",
    "1. 2014-2018 某通信设备公司 Java 工程师，离职原因：个人发展；证明人：前主管 池明亮。",
    "2. 2018-2022 某电商平台高级工程师，期间主导订单系统重构；证明人：架构师 冯赛。",
    "3. 2022 至今 某金融科技公司技术专家；现居杭州市滨江区江汉路 178 号。",
    "三、信用核查（候选人授权）：名下有按揭房产一套，无失信记录，无重大诉讼。",
    "四、竞业限制核查：与原单位签订的竞业协议已于 2026 年 10 月到期。",
    "五、调查结论：未发现虚假履历与负面记录，建议正常录用。",
    "调查员：第三方背调机构 顾行；报告日期 2026 年 12 月 8 日。"),
  [("CN_NAME", "乐嘉树", True, "candidate_identity"),
   ("CN_NAME", "池明亮", True, "private_person"),
   ("CN_NAME", "冯赛", True, "private_person"),
   ("CN_ADDRESS", "杭州市滨江区江汉路 178 号", True, "private_address"),
   ("CN_NAME", "顾行", True, "private_person")],
  [("FINANCIAL", "名下有按揭房产一套", True, "asset_info"),
   ("EMPLOYMENT", "竞业协议已于 2026 年 10 月到期", True, "employment_restriction")],
  risk_group="quasi_identifier")


D("case_034", "zh", "long", "insurance_claim",
  P("人身保险理赔申请材料（示例）",
    "理赔申请书编号：CLM-2026-1208-009。",
    "一、出险人信息：姓名 温倩，性别女，出生日期 1979 年 8 月 3 日，身份证号 330106197908032267。",
    "二、保单信息：重大疾病保险，保单号 PO-2019-33882，基本保额 50 万元。",
    "三、出险情况：被保险人于 2026 年 11 月 20 日在浙江大学医学院附属某医院确诊为甲状腺乳头状癌，",
    "住院行甲状腺切除手术，住院 8 天，产生医疗费用 5.8 万元。",
    "四、就诊资料：病理报告（编号 PATH-2026-7712）、手术记录、出院小结各一份。",
    "五、受益人信息：受益人为配偶 谢东升，银行卡号 6222 0202 1000 3882 771。",
    "六、联系方式：被保险人手机 13777778888，家庭住址西湖区古墩路 701 号紫金广场 6 幢 1102 室。",
    "七、声明：本人承诺所提交材料真实，如有虚假愿承担法律责任。"),
  [("RECORD_ID", "CLM-2026-1208-009", True, "claim_reference"),
   ("CN_NAME", "温倩", True, "private_person"),
   ("CN_BIRTH_DATE", "1979 年 8 月 3 日", True, "birth_date"),
   ("CN_ID_CARD", "330106197908032267", True, "government_identity"),
   ("RECORD_ID", "PO-2019-33882", True, "policy_reference"),
   ("CN_NAME", "谢东升", True, "private_person"),
   ("CN_BANK_CARD", "6222 0202 1000 3882 771", True, "financial_instrument"),
   ("CN_PHONE_NUMBER", "13777778888", True, "private_contact"),
   ("CN_ADDRESS", "西湖区古墩路 701 号紫金广场 6 幢 1102 室", True, "private_address")],
  [("MEDICAL", "甲状腺乳头状癌", True, "diagnosis"),
   ("MEDICAL", "甲状腺切除手术", True, "procedure"),
   ("FINANCIAL", "医疗费用 5.8 万元", True, "medical_expense"),
   ("FINANCIAL", "基本保额 50 万元", True, "policy_amount")])

D("case_035", "zh", "long", "handover",
  P("项目交接文档（节选，内部资料）",
    "一、项目背景",
    "本项目为某省高速公路收费系统升级项目，合同编号 HT-2026-1188，项目经理 阮世杰。",
    "二、系统架构",
    "核心数据库为 PostgreSQL 15，部署于客户机房 A 区 3 号机柜，内网地址 10.60.12.20，",
    "连接串 postgres://toll_admin:Tr0ll_B00th#2026@10.60.12.20:5432/tollgate；",
    "应用服务器 4 台，负载均衡地址 10.60.12.30。",
    "三、供应商与对接人",
    "数据库厂商对接人：苏芮（电话 13800002222）；门禁系统厂商：李厚朴（电话 13800003333）。",
    "四、遗留问题",
    "1. 收费站 5 号车道抓拍相机固件存在已知 bug，补丁待厂商提供；",
    "2. 与高速集团的对账接口联调延期，集团侧联系人 童谣（邮箱 tongyao@expressway.example.cn）。",
    "五、移交清单",
    "密钥文件、部署手册、运维值班表已移交新任项目经理 康博闻。"),
  [("RECORD_ID", "HT-2026-1188", True, "contract_reference"),
   ("CN_NAME", "阮世杰", True, "private_person"),
   ("DATABASE_URI", "postgres://toll_admin:Tr0ll_B00th#2026@10.60.12.20:5432/tollgate", True, "database_credential"),
   ("IP_ADDRESS", "10.60.12.30", True, "internal_infrastructure"),
   ("CN_NAME", "苏芮", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800002222", True, "private_contact"),
   ("CN_NAME", "李厚朴", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800003333", True, "private_contact"),
   ("EMAIL", "tongyao@expressway.example.cn", True, "private_contact"),
   ("CN_NAME", "康博闻", True, "private_person")])

D("case_036", "zh", "long", "education",
  P("研究生复试记录（示例）",
    "一、考生信息：姓名 滕雨欣，本科毕业于河南某大学软件学院，报考专业：计算机技术。",
    "二、复试小组：组长 教授 华建；成员 副教授 雷鸣、企业导师 沙宝亮。",
    "三、面试问答记录（节选）",
    "问：请介绍你参与过的最复杂的项目。",
    "答：本科期间在导师 汪教授 指导下参与省重点实验室的舆情分析平台，负责数据采集模块，",
    "使用 Python 与 Kafka，日处理数据量约 2 亿条。",
    "备注：考生曾用名含生僻字 𠀀伟，已按规范登记。",
    "问：你的家庭住址与联系方式？",
    "答：现住郑州市金水区文化路 88 号院 7 号楼；手机 13522223333。",
    "四、初试成绩：政治 68，英语 72，数学 118，专业课 126，总分 384，排名第五。",
    "五、复试结论：同意拟录取，导师意向：雷鸣副教授。"),
  [("CN_NAME", "滕雨欣", True, "candidate_identity"),
   ("CN_NAME", "华建", True, "private_person"),
   ("CN_NAME", "雷鸣", True, "private_person"),
   ("CN_NAME", "沙宝亮", True, "private_person"),
   ("CN_NAME", "汪教授", True, "private_person"),
   ("CN_ADDRESS", "郑州市金水区文化路 88 号院 7 号楼", True, "private_address"),
   ("CN_PHONE_NUMBER", "13522223333", True, "private_contact")])

D("case_037", "zh", "long", "business_email_chain",
  P("邮件往来（合作洽谈，示例）",
    "第一封 12 月 03 日 09:12",
    "发件人：wang.ming@partner-co.example.cn",
    "收件人：li.xiaotong@our-startup.example.com",
    "主题：关于 12 月 18 日产品发布会的合作确认",
    "李晓童您好，我方确认出席发布会。随函附上我方参会展商资料：",
    "公司：上海某智能硬件有限公司；联系人：王明；手机 13911112222。",
    "第二封 12 月 03 日 14:30",
    "发件人：li.xiaotong@our-startup.example.com",
    "主题：回复：关于 12 月 18 日产品发布会的合作确认",
    "王总您好，已收到资料。展位布展负责人为我方同事 方之遥，联系电话 13722223333，",
    "布展地址：浦东新区世博大道 1000 号 3 号馆 B12 展位。",
    "第三封 12 月 05 日 10:05",
    "发件人：wang.ming@partner-co.example.cn",
    "主题：发票信息",
    "我司开票信息：上海某智能硬件有限公司，税号 91310000MA1FL88213，",
    "开户行：招商银行上海分行营业部，账号 1219 0288 2100 88213。"),
  [("EMAIL", "wang.ming@partner-co.example.cn", True, "private_contact"),
   ("EMAIL", "li.xiaotong@our-startup.example.com", True, "private_contact"),
   ("CN_NAME", "李晓童", True, "private_person"),
   ("CN_NAME", "王明", True, "private_person"),
   ("CN_PHONE_NUMBER", "13911112222", True, "private_contact"),
   ("CN_NAME", "方之遥", True, "private_person"),
   ("CN_PHONE_NUMBER", "13722223333", True, "private_contact"),
   ("CN_ADDRESS", "浦东新区世博大道 1000 号 3 号馆 B12 展位", True, "business_location"),
   ("RECORD_ID", "91310000MA1FL88213", True, "tax_identifier")])

D("case_038", "zh", "long", "quasi_patient",
  P("社区健康档案（示例）",
    "档案编号：HRA-2026-0119。",
    "基本信息：姓名 屠维，男，67 岁；文化程度：初中；婚姻状况：丧偶。",
    "既往史：高血压 15 年；冠心病 8 年；前列腺增生 3 年。",
    "生活史：吸烟 40 年，每日 10 支；饮酒已戒 5 年。",
    "家族史：父亲因脑卒中去世；兄患 2 型糖尿病。",
    "现用药：阿司匹林肠溶片、阿托伐他汀钙片、特拉唑嗪。",
    "家庭住址：宁波市鄞州区首南街道某小区 11 幢 302 室。",
    "签约家庭医生：社区服务中心 全科医生 水冰心；随访计划：每季度一次。",
    "紧急联系人：女儿 屠春晓，电话 13400005555。"),
  [("RECORD_ID", "HRA-2026-0119", True, "health_record_reference"),
   ("CN_NAME", "屠维", True, "patient_identity"),
   ("CN_NAME", "水冰心", True, "private_person"),
   ("CN_NAME", "屠春晓", True, "private_person"),
   ("CN_PHONE_NUMBER", "13400005555", True, "private_contact")],
  [("MEDICAL", "高血压 15 年", True, "medical_history"),
   ("MEDICAL", "冠心病 8 年", True, "medical_history"),
   ("MEDICAL", "前列腺增生 3 年", True, "medical_history"),
   ("MEDICAL", "父亲因脑卒中去世", True, "family_history"),
   ("MEDICAL", "阿司匹林肠溶片、阿托伐他汀钙片", True, "medication")],
  risk_group="quasi_identifier")

D("case_039", "zh", "very_long", "court_fiction",
  P("（模拟法庭教学案例，全部人物与机构均为虚构）",
    "一、案件基本情况",
    "原告：满志刚，男，1975 年 3 月 12 日出生，汉族，个体经营者，住苏州市姑苏区平江路 248 号。",
    "被告：苏州某餐饮管理有限公司，住所地苏州市工业园区星湖街 328 号。",
    "案由：劳务合同纠纷。",
    "二、原告诉称",
    "原告自 2023 年 3 月起受雇于被告经营的某连锁餐馆，担任厨师长，月工资 12,000 元。",
    "2026 年 9 月 30 日，被告单方面解除劳务关系，拖欠原告 2026 年 8 月至 9 月工资合计 24,000 元，",
    "且未支付未休年假折算工资 3,310 元。原告多次催讨未果，故诉至法院。",
    "三、被告辩称",
    "被告辩称，原告在职期间多次违反食品卫生操作规范，2026 年 8 月 15 日因冷库温度记录造假被通报，",
    "被告依据双方签订的劳务协议第 7 条第 2 款解除协议合法有据，不存在拖欠工资情形。",
    "四、证据清单",
    "原告提交：银行流水（尾号 6677 卡号 6222 0208 0000 6677 432）、微信催讨记录、考勤截图。",
    "被告提交：劳务协议、卫生检查通报（编号 HC-2026-088）、员工手册签收单。",
    "五、法院查明",
    "经审理查明，原告确实存在 2026 年 8 月 15 日冷库温度记录不规范行为，但被告未能提供按制度",
    "解除协议的工会通知程序证据；被告已支付原告 2026 年 8 月工资 12,000 元，9 月工资未付。",
    "六、裁判结果（模拟）",
    "一、被告于本判决生效之日起十日内支付原告工资 12,000 元；",
    "二、驳回原告其他诉讼请求。",
    "审判员（模拟）：吴中亮；书记员（模拟）：傅清清。",
    "如不服本判决，可在判决书送达之日起十五日内上诉于苏州市中级人民法院。",
    "本案纯属教学模拟，任何人物、机构、编号均非真实。"),
  [("CN_NAME", "满志刚", True, "private_person"),
   ("CN_BIRTH_DATE", "1975 年 3 月 12 日", True, "birth_date"),
   ("CN_ADDRESS", "苏州市姑苏区平江路 248 号", True, "private_address"),
   ("CN_BANK_CARD", "6222 0208 0000 6677 432", True, "financial_instrument")],
  [("FINANCIAL", "月工资 12,000 元", True, "income_info"),
   ("FINANCIAL", "工资合计 24,000 元", True, "debt_info"),
   ("EMPLOYMENT", "被告单方面解除劳务关系", True, "employment_dispute")])

D("case_040", "zh", "very_long", "merger_memo",
  P("并购项目备忘录（内部机密，示例）",
    "密级：机密；项目代号：灯塔；参与方：甲方 某新能源集团，乙方 某电池材料有限公司。",
    "一、交易结构",
    "甲方拟以现金加换股方式收购乙方 65% 股权，交易对价预估 18.6 亿元，最终以评估报告为准。",
    "二、尽职调查范围",
    "1. 乙方主体资格、股权结构（股东名册见附件一：创始人 焦立群持股 48%，某创投基金持股 30%，",
    "员工持股平台持股 22%）；",
    "2. 乙方主要资产：常州生产基地（地址 常州市金坛区中兴路 88 号）、2 项发明专利（ZL202610882139.5、",
    "ZL202610882140.X）；",
    "3. 乙方财务：截至 2026 年 11 月 30 日，账面净资产 7.2 亿元，银行借款余额 3.1 亿元，",
    "其中一笔 8,000 万元流动资金贷款将于 2027 年 3 月到期；",
    "4. 重大合同：与某车企签订的长期供货协议（合同号 LTA-2026-0309），违约金条款需重点核查。",
    "三、人员安排",
    "交割后乙方创始人 焦立群 继续担任总经理三年；财务负责人由甲方委派 田知远 接任。",
    "四、时间表",
    "2027 年 1 月 15 日前签署正式协议；2027 年 2 月 28 日前完成反垄断申报。",
    "五、项目组联系方式",
    "甲方项目负责人：投资部总监 元振海（邮箱 yuanzhenhai@newenergy-group.example.com）；",
    "乙方授权代表：董事会秘书 秋水（电话 13799996666）。",
    "六、保密与文件管理",
    "本备忘录仅限项目组传阅；文件存放于甲方内网资料室，访问地址 https://vault.newenergy-group.example.com/lighthouse，",
    "账号口令由 IT 统一发放，禁止转发至个人邮箱。"),
  [("CN_NAME", "焦立群", True, "private_person"),
   ("CN_NAME", "田知远", True, "private_person"),
   ("CN_NAME", "元振海", True, "private_person"),
   ("CN_NAME", "秋水", True, "private_person"),
   ("CN_PHONE_NUMBER", "13799996666", True, "private_contact"),
   ("EMAIL", "yuanzhenhai@newenergy-group.example.com", True, "private_contact"),
   ("CN_ADDRESS", "常州市金坛区中兴路 88 号", True, "business_location")],
  [("FINANCIAL", "交易对价预估 18.6 亿元", True, "deal_amount"),
   ("FINANCIAL", "银行借款余额 3.1 亿元", True, "debt_info"),
   ("FINANCIAL", "8,000 万元流动资金贷款", True, "debt_info"),
   ("COMMERCIAL_SECRET", "项目代号：灯塔", True, "codename")])


# ---------------------------------------------------------------------------
# case_041-050: Chinese long (rest)
# ---------------------------------------------------------------------------

D("case_041", "zh", "long", "relocation",
  P("移民留学咨询记录（示例）",
    "客户：贺兰山，42 岁，与妻子 石榴，39 岁，育有一子 贺小舟，8 岁。",
    "咨询诉求：全家技术移民评估，目标国家：加拿大。",
    "主申请人语言成绩：雅思总分 7.0（听力 7.5，阅读 7.5，写作 6.5，口语 6.0）。",
    "学历认证：master 学位，专业 计算机科学；职业代码 NOC 21231。",
    "资产证明：活期存款 80 万元；名下房产一套（评估价 420 万元），贷款余额 160 万元。",
    "顾问：某移民公司资深顾问 那英群；客户经理邮箱 na.yingqun@immigrate.example.com。",
    "客户联系电话：13800004444；现住址：天津市和平区南京路 200 号 3 幢 902 室。"),
  [("CN_NAME", "贺兰山", True, "private_person"),
   ("CN_NAME", "石榴", True, "private_person"),
   ("CN_NAME", "贺小舟", True, "private_person"),
   ("CN_NAME", "那英群", True, "private_person"),
   ("EMAIL", "na.yingqun@immigrate.example.com", False, "corporate_role_mailbox"),
   ("CN_PHONE_NUMBER", "13800004444", True, "private_contact"),
   ("CN_ADDRESS", "天津市和平区南京路 200 号 3 幢 902 室", True, "private_address")],
  [("FINANCIAL", "活期存款 80 万元", True, "asset_info"),
   ("FINANCIAL", "贷款余额 160 万元", True, "debt_info")],
  risk_group="quasi_identifier")


D("case_042", "zh", "long", "manufacturing",
  P("设备故障处理报告（示例）",
    "一、故障概况",
    "设备：3 号注塑机（品牌：某德系品牌，型号 VA-880）；故障时间：12 月 11 日 14:22。",
    "报修人：车间班长 越鹏；维修单号 REP-2026-1208。",
    "二、故障现象",
    "合模机构异响，锁模力波动 ±8%，产品飞边率上升至 6%。",
    "三、处理过程",
    "14:40 机修工 简大成 到场排查；15:20 确认锁模油缸密封件老化；",
    "16:05 更换密封件并校准压力；17:30 试模 50 模次合格，恢复正常生产。",
    "四、备件与费用",
    "更换密封件 2 套（单价 860 元），工时 3.5 小时，合计费用 4,310 元。",
    "五、预防措施",
    "建议将液压系统保养周期由 3,000 小时缩短至 2,500 小时，责任部门：设备部。",
    "六、相关方联系信息",
    "设备厂商售后热线 400-880-1234；区域工程师 佟丽（13800005555）；",
    "厂内备件库管理员 桑榆（内线电话 8021）。"),
  [("CN_NAME", "越鹏", True, "private_person"),
   ("RECORD_ID", "REP-2026-1208", True, "repair_reference"),
   ("CN_NAME", "简大成", True, "private_person"),
   ("PHONE", "400-880-1234", False, "business_hotline"),
   ("CN_NAME", "佟丽", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800005555", True, "private_contact")],
  [("FINANCIAL", "合计费用 4,310 元", True, "maintenance_cost")])

D("case_043", "zh", "long", "community_epidemic",
  P("社区卫生服务通知（示例）",
    "各位居民：",
    "根据区卫健部门统一安排，本社区将于 12 月 26 日至 27 日开展重点人群健康随访。",
    "随访对象：65 岁以上老年人、慢性病患者、孕产妇。",
    "随访内容：血压血糖测量、用药指导、疫苗接种提醒。",
    "请以下登记居民配合：",
    "1 号楼：常山（高血压）、仰月（糖尿病随访）；",
    "2 号楼：米乐夫妇（孕 32 周产检提醒）；",
    "3 号楼：山丹（长期卧床，上门随访）。",
    "随访医护：全科团队 王医生、护士 廉清；联系电话 0571-88990011。",
    "家庭医生签约咨询：社区卫生服务中心一楼 105 室。",
    "本通知所涉个人健康信息仅用于本次随访，禁止外传。"),
  [("CN_NAME", "常山", True, "private_person"),
   ("CN_NAME", "仰月", True, "private_person"),
   ("CN_NAME", "米乐", True, "private_person"),
   ("CN_NAME", "山丹", True, "private_person"),
   ("CN_NAME", "王医生", True, "private_person"),
   ("CN_NAME", "廉清", True, "private_person"),
   ("CN_LANDLINE", "0571-88990011", False, "business_hotline")],
  [("MEDICAL", "血压血糖测量、用药指导", True, "care_activity"),
   ("MEDICAL", "孕 32 周产检提醒", True, "pregnancy_info"),
   ("MEDICAL", "高血压", True, "chronic_condition"),
   ("MEDICAL", "糖尿病随访", True, "chronic_condition")],
  risk_group="quasi_identifier")

D("case_044", "zh", "very_long", "it_incident",
  P("信息安全事件报告（示例）",
    "一、事件概述",
    "2026 年 12 月 14 日 20:18，安全运营中心监测到生产环境数据库服务器 db-prod-03（内网地址",
    "10.20.0.33）出现异常批量导出行为，累计导出客户表数据约 40 万行，触发 DLP 告警。",
    "二、处置时间线",
    "20:19 SOC 值班分析师 尹航 确认告警为真实导出行为，立即电话通知数据库管理员 麦冬；",
    "20:25 麦冬冻结涉事账号 report_svc；",
    "20:31 确认导出来源为主办公楼 3 层研发网段终端 10.20.8.117，使用人为数据分析师 岑参；",
    "20:40 信息安全部经理 黎明 启动应急预案，隔离涉事终端；",
    "21:10 涉事终端完成镜像取证，移交法务部；",
    "12 月 15 日 10:00 与涉事员工面谈，其承认将客户手机号与姓名导出用于个人接洽的第三方营销。",
    "三、影响评估",
    "泄露字段：客户姓名、手机号、签约套餐；不涉及身份证号、银行卡与密码。",
    "涉及客户约 12 万人，需按监管要求完成通报与告知。",
    "四、整改措施",
    "1. 收紧报表系统导出权限，启用双人复核；",
    "2. 生产库敏感字段脱敏展示（手机号中间四位掩码）；",
    "3. 全员隐私合规培训，12 月 30 日前完成；",
    "4. 数据库审计日志保留期由 90 天延长至 180 天。",
    "五、责任与联系",
    "内部调查组：法务部 池鱼、人力资源部 蓝盈、信息安全部 尹航。",
    "监管报备联系人：合规部 潘岳，电话 13766663333，邮箱 panyue@corp.example.com。",
    "客户咨询专线 400-900-1234（工作日 9:00-18:00）。"),
  [("IP_ADDRESS", "10.20.0.33", True, "internal_infrastructure"),
   ("IP_ADDRESS", "10.20.8.117", True, "internal_infrastructure"),
   ("CN_NAME", "尹航", True, "private_person"),
   ("CN_NAME", "麦冬", True, "private_person"),
   ("CN_NAME", "岑参", True, "private_person"),
   ("CN_NAME", "黎明", True, "private_person"),
   ("CN_NAME", "池鱼", True, "private_person"),
   ("CN_NAME", "蓝盈", True, "private_person"),
   ("CN_NAME", "潘岳", True, "private_person"),
   ("CN_PHONE_NUMBER", "13766663333", True, "private_contact"),
   ("EMAIL", "panyue@corp.example.com", True, "private_contact"),
   ("PHONE", "400-900-1234", False, "business_hotline")],
  [("COMMERCIAL_SECRET", "泄露字段：客户姓名、手机号、签约套餐", True, "incident_scope")])

D("case_045", "zh", "very_long", "tender",
  P("招标项目联系人册（示例）",
    "项目名称：某市地铁 6 号线自动售检票系统采购项目；招标编号：METRO6-AFC-2026。",
    "一、招标人",
    "某市轨道交通集团有限公司；地址：某市江东区中山东路 1888 号轨道交通大厦。",
    "招标联系人：采购部 主任工程师 应长征，电话 0574-88112233；",
    "技术联系人：通号部 高工 明月心，邮箱 mingyuexin@metro6.example.cn。",
    "二、投标保证金与文件递交",
    "保证金 80 万元，于开标前两个工作日到账；递交地址：轨道交通大厦 3 层开标室。",
    "三、潜在投标人（资格预审通过名单）",
    "1. 某智能科技有限公司：授权代表 华笙，手机 13800006666，邮箱 huasheng@afc-vendor.example.com；",
    "2. 某电子信息股份公司：授权代表 糜信，手机 13800007777；",
    "3. 某科技集团：授权代表 公孙离，手机 13800008888。",
    "四、评标办法",
    "综合评分法：技术 60 分，商务 20 分，价格 20 分；评标委员会共 5 人（名单开标前保密）。",
    "五、关键时间节点",
    "答疑截止：2027 年 1 月 8 日 17:00；开标：2027 年 1 月 20 日 09:30。",
    "六、监管与投诉",
    "行政监督部门：某市公共资源交易监管局；投诉受理：0574-88334455。",
    "七、声明",
    "本联系人册仅用于本项目沟通，联系人个人信息受招标法及个保法保护，禁止用于其他用途。"),
  [("RECORD_ID", "METRO6-AFC-2026", True, "tender_reference"),
   ("CN_NAME", "应长征", True, "private_person"),
   ("CN_LANDLINE", "0574-88112233", False, "business_hotline"),
   ("CN_NAME", "明月心", True, "private_person"),
   ("EMAIL", "mingyuexin@metro6.example.cn", False, "corporate_role_mailbox"),
   ("CN_NAME", "华笙", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800006666", True, "private_contact"),
   ("EMAIL", "huasheng@afc-vendor.example.com", True, "private_contact"),
   ("CN_NAME", "糜信", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800007777", True, "private_contact"),
   ("CN_NAME", "公孙离", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800008888", True, "private_contact"),
   ("CN_LANDLINE", "0574-88334455", False, "business_hotline")])

D("case_046", "zh", "long", "startup_team",
  P("创业团队融资资料（节选，示例）",
    "一、团队核心成员",
    "CEO：柏舟，34 岁，连续创业者，曾创办某内容社区并成功退出；",
    "CTO：祈风，31 岁，前某大厂推荐算法负责人；",
    "COO：跃如，29 岁，前咨询公司项目经理。",
    "二、期权池与股权",
    "创始团队持股 70%，期权池 15%，天使投资人持股 15%；员工持股平台有限合伙注册于宁波梅山。",
    "三、本轮融资",
    "拟融资 3,000 万元人民币，出让 12% 股权，投前估值 2.2 亿元。",
    "资金用途：研发 60%，市场 25%，运营 15%。",
    "四、联系方式",
    "BP 接收邮箱：bp@startup-xyz.example.com；CEO 助理微信：ceobaizhou2026。",
    "办公地址：杭州市余杭区梦想小镇天使村 7 幢。",
    "五、声明",
    "本资料含商业秘密，接收方负有保密义务；文档水印编号 VW-2026-1219-88。"),
  [("CN_NAME", "柏舟", True, "private_person"),
   ("CN_NAME", "祈风", True, "private_person"),
   ("CN_NAME", "跃如", True, "private_person"),
   ("EMAIL", "bp@startup-xyz.example.com", False, "corporate_role_mailbox"),
   ("CN_SOCIAL_ACCOUNT", "ceobaizhou2026", True, "social_handle"),
   ("CN_ADDRESS", "杭州市余杭区梦想小镇天使村 7 幢", True, "business_location")],
  [("FINANCIAL", "拟融资 3,000 万元人民币", True, "funding_info"),
   ("FINANCIAL", "投前估值 2.2 亿元", True, "valuation_info"),
   ("COMMERCIAL_SECRET", "文档水印编号 VW-2026-1219-88", True, "document_watermark")])

D("case_047", "zh", "long", "quasi_athlete",
  P("青少年体育注册信息表（示例）",
    "运动员姓名：褚天翊；性别：男；出生日期：2012 年 6 月 15 日。",
    "注册项目：游泳（自由泳）；运动员注册号：ZJ-SW-2026-0882。",
    "就读学校：宁波市某实验学校六年级；班主任：何老师。",
    "训练履历：2021 年开始系统训练；现隶属 某市青少年游泳俱乐部；教练：魏武。",
    "最好成绩：50 米自由泳 27 秒 88（2026 年市青少年锦标赛 U12 组第一名）。",
    "监护人：父亲 褚建国，手机 13666667777；母亲 阎萍。",
    "家庭住址：宁波市鄞州区首南中路 500 号某小区 6 幢 1101 室。",
    "体检情况：心电图正常，无哮喘病史；既往肩部拉伤已痊愈。",
    "注册有效期至 2027 年 12 月 31 日。"),
  [("CN_NAME", "褚天翊", True, "minor_identity"),
   ("CN_BIRTH_DATE", "2012 年 6 月 15 日", True, "birth_date"),
   ("RECORD_ID", "ZJ-SW-2026-0882", True, "athlete_registration"),
   ("CN_NAME", "魏武", True, "private_person"),
   ("CN_NAME", "褚建国", True, "private_person"),
   ("CN_NAME", "阎萍", True, "private_person"),
   ("CN_PHONE_NUMBER", "13666667777", True, "private_contact"),
   ("CN_ADDRESS", "宁波市鄞州区首南中路 500 号某小区 6 幢 1101 室", True, "private_address")],
  [("MEDICAL", "心电图正常，无哮喘病史", False, "clean_health_summary"),
   ("MEDICAL", "既往肩部拉伤已痊愈", True, "injury_history")],
  risk_group="quasi_identifier",
  notes="minor athlete: school+club+results+home address combination")

D("case_048", "zh", "long", "government_service",
  P("政务窗口办件回执（示例）",
    "办件编号：ZJ-2026-1220-0088。",
    "办理事项：不动产转移登记（二手房）。",
    "申请人：卖方 闻人夏；买方 端木青。",
    "标的：下城区朝晖路 121 号 4 幢 2 单元 302 室及储藏间一间。",
    "缴税情况：契税 32,400 元已缴讫（票据号 FP-2026-88123999）。",
    "领取方式：邮寄（收件人 端木青，电话 13900001234，地址同标的房产）。",
    "承诺办结时限：5 个工作日；窗口电话 0571-12345 转 6。",
    "评价渠道：办结后可通过短信链接评价；监督电话 0571-96345。"),
  [("RECORD_ID", "ZJ-2026-1220-0088", True, "case_reference"),
   ("CN_NAME", "闻人夏", True, "private_person"),
   ("CN_NAME", "端木青", True, "private_person"),
   ("CN_ADDRESS", "下城区朝晖路 121 号 4 幢 2 单元 302 室", True, "property_address"),
   ("RECORD_ID", "FP-2026-88123999", True, "tax_receipt_reference"),
   ("CN_PHONE_NUMBER", "13900001234", True, "private_contact"),
   ("CN_LANDLINE", "0571-96345", False, "business_hotline")],
  [("FINANCIAL", "契税 32,400 元", True, "tax_amount")])

D("case_049", "zh", "long", "media_interview",
  P("人物专访速记（示例，受访者均为虚构）",
    "受访者：电声乐队「夜航西飞」主唱 黎洛；经纪人：唐difficulty".replace("唐difficulty", "唐阙"),
    "采访时间：12 月 15 日 15:00；地点：livehouse 后台。",
    "要点记录：",
    "1. 新专辑《午夜电台》将于次年 3 月发行，主打歌已进入混音阶段；",
    "2. 乐队经纪约明年到期，正在与两家唱片公司接触；",
    "3. 主唱近日声带小结，遵医嘱禁声两周，部分巡演延期；",
    "4. 商务合作对接：经纪人唐阙 13700002222，商务邮箱 booking@nightflight.example.com。",
    "速记整理：实习记者 乐正菱；审核：主编 蔚蓝。"),
  [("CN_NAME", "黎洛", True, "private_person"),
   ("CN_NAME", "唐阙", True, "private_person"),
   ("CN_PHONE_NUMBER", "13700002222", True, "private_contact"),
   ("EMAIL", "booking@nightflight.example.com", False, "corporate_role_mailbox"),
   ("CN_NAME", "乐正菱", True, "private_person"),
   ("CN_NAME", "蔚蓝", True, "private_person")],
  [("MEDICAL", "声带小结", True, "health_condition")])

D("case_050", "zh", "long", "senior_care",
  P("养老机构入住评估表（示例）",
    "一、长者信息：姓名 卫子安，性别 男，86 岁；退休前职业：中学物理教师。",
    "二、健康状况：阿尔茨海默病早期；高血压；听力下降（右耳佩戴助听器）。",
    "三、生活能力评估：ADL 评分 65 分，需协助洗浴与用药管理；跌倒风险：中度。",
    "四、用药清单：多奈哌齐片、氨氯地平片、钙剂。",
    "五、紧急联系人：长子 卫东升 13822223333；次女 卫南枝 13822224444。",
    "六、入住房间：颐养楼 2 层 208 房（朝南）；护理等级：二级护理。",
    "七、费用：床位费 4,800 元/月，护理费 3,600 元/月，餐费 1,200 元/月。",
    "八、评估员：护理部主任 庹晓岚；评估日期 2026 年 12 月 16 日。",
    "九、特别约定：外出需家属陪同；药物由机构医务室统一保管分发。"),
  [("CN_NAME", "卫子安", True, "elder_identity"),
   ("CN_NAME", "卫东升", True, "private_person"),
   ("CN_PHONE_NUMBER", "13822223333", True, "private_contact"),
   ("CN_NAME", "卫南枝", True, "private_person"),
   ("CN_PHONE_NUMBER", "13822224444", True, "private_contact"),
   ("CN_NAME", "庹晓岚", True, "private_person")],
  [("MEDICAL", "阿尔茨海默病早期", True, "diagnosis"),
   ("MEDICAL", "多奈哌齐片、氨氯地平片", True, "medication"),
   ("MEDICAL", "跌倒风险：中度", True, "risk_assessment"),
   ("FINANCIAL", "床位费 4,800 元/月", True, "care_cost")],
  risk_group="quasi_identifier")


# ---------------------------------------------------------------------------
# case_051-070: English short / medium
# ---------------------------------------------------------------------------

D("case_051", "en", "short", "account_config",
  "Username: Emily Carter",
  [("USERNAME", "Emily Carter", True, "account_handle")],
  notes="HARD CASE: explicit account context makes this USERNAME, not PERSON")

D("case_052", "en", "short", "meeting_note",
  "Emily Carter joined the meeting this morning. \U0001F5A5\uFE0F",
  [("PERSON", "Emily Carter", True, "private_person")],
  notes="HARD CASE: narrative mention is PERSON, never USERNAME")

D("case_053", "en", "short", "support_chat",
  "Agent: Could you confirm the phone number on file? Customer: (628) 555-0142 — texted a \U0001F4E6 label photo.",
  [("PHONE", "(628) 555-0142", True, "private_contact")])

D("case_054", "en", "short", "it_note",
  "The office printer is out of toner again. Please contact IT at 415-555-0199.",
  [("PHONE", "415-555-0199", False, "business_hotline")])

D("case_055", "en", "short", "doctor_note",
  "Patient Mason Reed reports persistent migraines; prescribed propranolol 40mg daily.",
  [("PERSON", "Mason Reed", True, "patient_identity")],
  [("MEDICAL", "persistent migraines", True, "symptom"),
   ("MEDICAL", "propranolol 40mg", True, "medication")])

D("case_056", "en", "short", "bank_letter",
  "Your checking account ending in 4471 was credited with $1,250.00 on Dec 18.",
  [("ACCOUNT_NUMBER", "4471", True, "financial_instrument")],
  notes="masked account tail: partial identifier")

D("case_057", "en", "short", "delivery",
  "Package for Sarah O'Neill (Jose\u0301\u2019s building), leave at 87 Maple Street if not home. 🚚",
  [("PERSON", "Sarah O'Neill", True, "private_person"),
   ("ADDRESS", "87 Maple Street", True, "private_address")])

D("case_058", "en", "short", "newsletter",
  "Subscribe with your email: john.doe@example.com to receive weekly digests.",
  [("EMAIL", "john.doe@example.com", False, "documentation_example")],
  notes="example.com placeholder: detect, do not redact")

D("case_059", "en", "short", "hotel",
  "Reservation confirmed: Lucas Meyer, room 512, check-in Dec 24, booking #BK-88231.",
  [("PERSON", "Lucas Meyer", True, "private_person"),
   ("RECORD_ID", "#BK-88231", True, "booking_reference")])

D("case_060", "en", "short", "school_form",
  "Emergency contact for Ava Thompson: grandmother, phone 555-0187.",
  [("PERSON", "Ava Thompson", True, "minor_identity"),
   ("PHONE", "555-0187", True, "private_contact")])

D("case_061", "en", "medium", "it_ticket",
  P("Ticket #INC-2026-8841 - Password reset for contractor",
    "Requester: facilities manager Dan Brody.",
    "Contractor account: d.brody_facilities (expires Jan 15).",
    "Callback: (415) 555-0164 between 9am-5pm.",
    "Approved by: IT supervisor Renee Cole.",
    "Temporary password issued via phone: Fa11#W1nd0w",
    "Note: do not send credentials to personal email addresses."),
  [("RECORD_ID", "#INC-2026-8841", True, "ticket_reference"),
   ("PERSON", "Dan Brody", True, "private_person"),
   ("USERNAME", "d.brody_facilities", True, "account_handle"),
   ("PHONE", "(415) 555-0164", True, "private_contact"),
   ("PERSON", "Renee Cole", True, "private_person"),
   ("SECRET", "Fa11#W1nd0w", True, "temporary_credential")])

D("case_062", "en", "medium", "hr_note",
  P("HR file note - performance improvement plan (draft)",
    "Employee: Natalie Woods, hired Mar 2023, payroll ID EMP-44028.",
    "Manager: team lead Victor Hale.",
    "Summary: two consecutive quarters below target; PIP meeting scheduled Jan 6.",
    "HR business partner: Priya Nair (priya.nair@corp-hr.example.com).",
    "Confidential - do not forward."),
  [("PERSON", "Natalie Woods", True, "employee_identity"),
   ("EMPLOYEE_ID", "EMP-44028", True, "employee_record"),
   ("PERSON", "Victor Hale", True, "private_person"),
   ("PERSON", "Priya Nair", True, "private_person"),
   ("EMAIL", "priya.nair@corp-hr.example.com", True, "private_contact")],
  [("EMPLOYMENT", "two consecutive quarters below target", True, "performance_info"),
   ("EMPLOYMENT", "PIP meeting scheduled Jan 6", True, "employment_action")])

D("case_063", "en", "medium", "insurance_claim",
  P("Auto claim summary",
    "Claimant: Henry Walsh. Policy: AUTO-226-88123.",
    "Incident: rear-end collision on I-280 near exit 4, Dec 12, 17:40.",
    "Vehicle: 2021 silver sedan, plate 8ABC123.",
    "Estimated repair: $3,480. Deductible: $500.",
    "Adjuster: Marcus Lee, (408) 555-0171, extension 223."),
  [("PERSON", "Henry Walsh", True, "private_person"),
   ("RECORD_ID", "AUTO-226-88123", True, "policy_reference"),
   ("CN_LICENSE_PLATE", "8ABC123", True, "vehicle_identifier"),
   ("PERSON", "Marcus Lee", True, "private_person"),
   ("PHONE", "(408) 555-0171", True, "private_contact")],
  [("FINANCIAL", "$3,480", True, "claim_amount"),
   ("FINANCIAL", "$500", True, "deductible")])

D("case_064", "en", "medium", "job_application",
  P("Application snapshot - data analyst role",
    "Candidate: Grace Kim.",
    "Email: grace.kim.92@mail-example.com; Phone: (650) 555-0188.",
    "Address: 1420 Alumni Way, Apt 9, San Jose.",
    "Current title: BI analyst at a retail chain; notice period 4 weeks.",
    "References: former manager Alan Petrosian; colleague Yuki Tanaka."),
  [("PERSON", "Grace Kim", True, "candidate_identity"),
   ("EMAIL", "grace.kim.92@mail-example.com", True, "private_contact"),
   ("PHONE", "(650) 555-0188", True, "private_contact"),
   ("ADDRESS", "1420 Alumni Way, Apt 9, San Jose", True, "private_address"),
   ("PERSON", "Alan Petrosian", True, "private_person"),
   ("PERSON", "Yuki Tanaka", True, "private_person")])

D("case_065", "en", "medium", "patient_intake",
  P("Clinic intake form (example)",
    "Name: Omar Haddad. Date of birth: 04/17/1988.",
    "Allergies: penicillin (rash). Chronic conditions: none reported.",
    "Insurance member ID: MED-88-447102.",
    "Preferred pharmacy: the one on 3rd Street.",
    "Cell: (669) 555-0123."),
  [("PERSON", "Omar Haddad", True, "patient_identity"),
   ("CN_BIRTH_DATE", "04/17/1988", True, "birth_date"),
   ("MEDICAL", "penicillin (rash)", True, "allergy"),
   ("RECORD_ID", "MED-88-447102", True, "insurance_reference"),
   ("PHONE", "(669) 555-0123", True, "private_contact")],
  risk_group="quasi_identifier")


D("case_066", "en", "medium", "customer_complaint",
  P("Complaint log entry",
    "Customer: Brianna Cole (loyalty tier: gold).",
    "Order: 5521-8813, placed Dec 14, never delivered.",
    "Contact email given on the call: bric.88@webmail-example.net.",
    "Resolution offered: full refund of $84.20 plus a $15 voucher.",
    "Agent: Tom R. Escalated: no."),
  [("PERSON", "Brianna Cole", True, "private_person"),
   ("RECORD_ID", "5521-8813", True, "order_reference"),
   ("EMAIL", "bric.88@webmail-example.net", True, "private_contact"),
   ("PERSON", "Tom R.", True, "employee_identity")],
  [("FINANCIAL", "$84.20", True, "refund_amount")])

D("case_067", "en", "medium", "legal_letter",
  P("Letter of demand (example, fictional parties)",
    "To: Harmon & Gable LLP, 500 Court Street, Sacramento.",
    "From: counsel for Diana Frost.",
    "Re: unpaid invoice INV-2026-7781 in the amount of $12,400.",
    "Our client provided consulting services between Aug and Oct 2026.",
    "Payment is demanded within 14 days of this letter.",
    "Contact: attorney Jeremy Wu, (916) 555-0149."),
  [("ADDRESS", "500 Court Street, Sacramento", True, "business_location"),
   ("PERSON", "Diana Frost", True, "private_person"),
   ("RECORD_ID", "INV-2026-7781", True, "invoice_reference"),
   ("PERSON", "Jeremy Wu", True, "private_person"),
   ("PHONE", "(916) 555-0149", True, "private_contact")],
  [("FINANCIAL", "$12,400", True, "debt_info")])

D("case_068", "en", "medium", "gym_signup",
  P("Fitness membership agreement (excerpt)",
    "Member: Kyle Ramsay. Emergency contact: roommate Paul Nunez, (628) 555-0129.",
    "Plan: 12-month standard, $39/month, billed to card ending 8801.",
    "Medical waiver: member reports mild asthma; inhaler kept in locker 27.",
    "Start date: Jan 2. Club location: 88 Bayside Ave."),
  [("PERSON", "Kyle Ramsay", True, "private_person"),
   ("PERSON", "Paul Nunez", True, "private_person"),
   ("PHONE", "(628) 555-0129", True, "private_contact"),
   ("ACCOUNT_NUMBER", "8801", True, "financial_instrument"),
   ("MEDICAL", "mild asthma", True, "health_condition")])

D("case_069", "en", "medium", "quasi_academic",
  P("Reviewer profile (example, fictional person)",
    "Name: Prof. Elena Sokolova, age 45.",
    "Affiliation: department of robotics, a public university in Boston.",
    "Research:_slam_long_term_autonomy".replace("_slam_long_term_autonomy", " SLAM long-term autonomy."),
    "Lab: Room 3.212,建成后".replace("建成后", "Building C."),
    "Recent paper: Long-Term Mapping with Seasonal Drift (2025).",
    "Office phone: (617) 555-0132."),
  [("PERSON", "Elena Sokolova", True, "private_person"),
   ("PHONE", "(617) 555-0132", True, "private_contact")],
  risk_group="quasi_identifier",
  notes="age+affiliation+research+lab+paper combination")

D("case_070", "en", "medium", "airline",
  P("Flight disruption notice",
    "Passenger: Nathan Okafor. Booking: QF-88X21. Flight QA447 SIN-SYD delayed 6 hours.",
    "Rebooking confirmed on QA449, seat 31C.",
    "Meal voucher code: VCH-88213, value $18.",
    "Baggage claim tag: 0298821.",
    "Duty manager hotline: +65 6555 0177."),
  [("PERSON", "Nathan Okafor", True, "private_person"),
   ("RECORD_ID", "QF-88X21", True, "booking_reference"),
   ("RECORD_ID", "VCH-88213", True, "voucher_reference"),
   ("RECORD_ID", "0298821", True, "baggage_tag"),
   ("PHONE", "+65 6555 0177", False, "business_hotline")])


# ---------------------------------------------------------------------------
# case_071-080: English long (5 long + 3 very_long + 2 long)
# ---------------------------------------------------------------------------

D("case_071", "en", "long", "soc_report",
  P("Statement of caregiver duties - home care (example)",
    "Client: Margaret Liu, 79, living alone at 12 Birchwood Lane.",
    "Care plan effective Dec 1 through Feb 28.",
    "Monday/Wednesday/Friday: medication reminders (lisinopril, metformin),",
    "light housekeeping, blood pressure log.",
    "Tuesday/Thursday: physiotherapy escort to the community center,",
    "fall-risk assessment after the November hip scare.",
    "Saturday: grocery run with the aide; budget $60 per week.",
    "Primary aide: Rosa Vargas. Backup: James Chen.",
    "Emergency contacts: daughter Amy Liu (510) 555-0166;",
    "neighbor Peter Alvarez (510) 555-0167 has a spare key.",
    "Care coordinator: Diana Ross-Wellington, monthly review calls.",
    "Client prefers no male aides after 6pm; service dog Buddy is on site."),
  [("PERSON", "Margaret Liu", True, "elder_identity"),
   ("ADDRESS", "12 Birchwood Lane", True, "private_address"),
   ("MEDICAL", "lisinopril, metformin", True, "medication"),
   ("PERSON", "Rosa Vargas", True, "private_person"),
   ("PERSON", "James Chen", True, "private_person"),
   ("PERSON", "Amy Liu", True, "private_person"),
   ("PHONE", "(510) 555-0166", True, "private_contact"),
   ("PERSON", "Peter Alvarez", True, "private_person"),
   ("PHONE", "(510) 555-0167", True, "private_contact"),
   ("PERSON", "Diana Ross-Wellington", True, "private_person")],
  [("MEDICAL", "fall-risk assessment", True, "risk_assessment")])

D("case_072", "en", "long", "project_handover",
  P("Handover notes - payments platform (example)",
    "Outgoing lead: Priya Raghavan. Incoming lead: Tomasz Kowalski.",
    "Architecture: three API pods behind an internal ALB (10.44.1.20),",
    "PostgreSQL 14 primary at 10.44.1.31 with a streaming replica at 10.44.1.32.",
    "Connection string for the replica pool:",
    "postgres://reader_svc:R3ad0nly!2026@10.44.1.32:5432/payments_replica",
    "Third-party integrations: PSP A (webhook secret rotated monthly,",
    "stored in the vault under keys/payments/psp_a), PSP B (sandbox only).",
    "Open incidents: INC-7712 (idempotency gaps on retries, owner Tomasz),",
    "INC-7718 (reconciliation report slow at month-end, owner Anita Deshmukh).",
    "Compliance: PCI scope refresh due Feb; QSA contact Fiona Braithwaite.",
    "Runbook location: the internal wiki, restricted to the payments group.",
    "Escalation: VP Engineering Nadia Osei.",
    "Do not commit credentials to the repository; use the vault."),
  [("PERSON", "Priya Raghavan", True, "private_person"),
   ("PERSON", "Tomasz Kowalski", True, "private_person"),
   ("IP_ADDRESS", "10.44.1.20", True, "internal_infrastructure"),
   ("DATABASE_URI", "postgres://reader_svc:R3ad0nly!2026@10.44.1.32:5432/payments_replica", True, "database_credential"),
   ("IP_ADDRESS", "10.44.1.31", True, "internal_infrastructure"),
   ("PERSON", "Anita Deshmukh", True, "private_person"),
   ("PERSON", "Fiona Braithwaite", True, "private_person"),
   ("PERSON", "Nadia Osei", True, "private_person")])

D("case_073", "en", "long", "patient_discharge",
  P("Discharge summary (example, fictional patient)",
    "Patient: George Whitfield, male, DOB 02/03/1955, MRN 0088213.",
    "Admission: Dec 2, community-acquired pneumonia, right lower lobe.",
    "Hospital course: treated with IV ceftriaxone and azithromycin;",
    "oxygen weaned from 4L to room air by day 5.",
    "Comorbidities: COPD (GOLD II), type 2 diabetes on metformin 1000mg BID.",
    "Discharge meds: amoxicillin-clavulanate 875/125 BID x7 days,",
    "prednisone 40mg daily x5 days taper, continue inhalers.",
    "Follow-up: chest clinic in 2 weeks; GP review in 7 days.",
    "Warn-and-return advice provided to daughter Helen Whitfield,",
    "contact (206) 555-0188.",
    "Discharging physician: Dr. Amara Nwosu."),
  [("PERSON", "George Whitfield", True, "patient_identity"),
   ("RECORD_ID", "0088213", True, "medical_record_reference"),
   ("PERSON", "Helen Whitfield", True, "private_person"),
   ("PHONE", "(206) 555-0188", True, "private_contact"),
   ("PERSON", "Amara Nwosu", True, "private_person")],
  [("MEDICAL", "community-acquired pneumonia", True, "diagnosis"),
   ("MEDICAL", "COPD (GOLD II)", True, "comorbidity"),
   ("MEDICAL", "type 2 diabetes", True, "comorbidity"),
   ("MEDICAL", "ceftriaxone and azithromycin", True, "medication"),
   ("MEDICAL", "prednisone 40mg daily x5 days taper", True, "medication")])

D("case_074", "en", "long", "school_incident",
  P("Incident report - school field trip (example)",
    "School: Lakeside Middle School. Trip: science museum, Dec 16.",
    "Students involved: four eighth-graders; names withheld from this copy,",
    "see appendix held by the vice principal.",
    "Chaperones: Mr. Daniel Pierce, Ms. Yolanda Reyes, parent volunteer Mrs. Kim.",
    "Incident: one student slipped on the lobby stairs at 11:20,",
    "suspected sprained wrist; first aid applied; parents called at 11:35.",
    "Guardian contact used: father, phone (425) 555-0175.",
    "Transport: student picked up by guardian at 12:10.",
    "Follow-up: incident form 2026-TRIP-119 filed; museum safety officer notified.",
    "Reviewed by: Principal Sandra Whitmore."),
  [("PERSON", "Daniel Pierce", True, "private_person"),
   ("PERSON", "Yolanda Reyes", True, "private_person"),
   ("PHONE", "(425) 555-0175", True, "private_contact"),
   ("RECORD_ID", "2026-TRIP-119", True, "incident_reference"),
   ("PERSON", "Sandra Whitmore", True, "private_person")],
  [("MEDICAL", "suspected sprained wrist", True, "injury")])

D("case_075", "en", "long", "landlord_tenant",
  P("Move-out inspection record (example)",
    "Property: Unit 4B, 220 Harbor View Apartments.",
    "Tenant: Omar El-Sayed. Lease ended Nov 30.",
    "Forwarding address provided: 88 Dock Road, Unit 12.",
    "Inspection findings: carpet stained in bedroom 2; two blinds broken;",
    "walls fair; kitchen appliances cleaned.",
    "Deposit: $1,800 held; proposed deductions $340 (carpet cleaning),",
    "$85 (blinds). Balance to refund: $1,375.",
    "Tenant disputes the carpet charge per email dated Dec 3;",
    "mediation offered Dec 18, landlord accepted.",
    "Property manager: Claire Devereux, (503) 555-0191."),
  [("ADDRESS", "Unit 4B, 220 Harbor View Apartments", True, "property_address"),
   ("PERSON", "Omar El-Sayed", True, "private_person"),
   ("ADDRESS", "88 Dock Road, Unit 12", True, "private_address"),
   ("PERSON", "Claire Devereux", True, "private_person"),
   ("PHONE", "(503) 555-0191", True, "private_contact")],
  [("FINANCIAL", "$1,800", True, "deposit_amount"),
   ("FINANCIAL", "$1,375", True, "refund_amount")])

D("case_076", "en", "very_long", "police_report_fiction",
  P("Fictional training scenario - not a real incident report",
    "All names, addresses and identifiers in this scenario are synthetic.",
    "On Dec 10 at approximately 22:15, officers responded to a report of a",
    "residential burglary in progress at 41 Copperfield Row.",
    "Reporting party: neighbor Walter Jensen, who called the non-emergency line.",
    "Upon arrival, officers met the resident, Ms. Ingrid Halvorsen,",
    "who stated she returned home at 22:05 and found the rear door forced.",
    "Missing items: one laptop (serial omitted from this copy),",
    "jewelry box contents unknown, approximately $300 in cash.",
    "No injuries reported. K9 unit conducted a track with negative results.",
    "Evidence technicians photographed tool marks on the door frame",
    "and collected two partial fingerprints from the kitchen window.",
    "Insurance details provided by the resident: policy HO-88213-2026.",
    "Case number: 2026-TR-088213. Reporting officer: Ofc. D. Marsh,",
    "badge 4417. Supervising sergeant: Sgt. L. Ortega.",
    "Follow-up scheduled with the resident by phone at (360) 555-0144.",
    "Community liaison will circulate a prevention notice to the block.",
    "Note: this document is a synthetic training artifact."),
  [("ADDRESS", "41 Copperfield Row", True, "incident_location"),
   ("PERSON", "Walter Jensen", True, "private_person"),
   ("PERSON", "Ingrid Halvorsen", True, "private_person"),
   ("RECORD_ID", "HO-88213-2026", True, "policy_reference"),
   ("RECORD_ID", "2026-TR-088213", True, "case_number"),
   ("PHONE", "(360) 555-0144", True, "private_contact"),
   ("PERSON", "Marsh", True, "employee_identity"),
   ("PERSON", "Ortega", True, "employee_identity")])

D("case_077", "en", "very_long", "audit",
  P("Internal audit excerpt - expense reconciliation (example)",
    "Scope: travel and entertainment expenses, Q3, sales division.",
    "Population: 412 reports; sample: 38 reports selected by risk.",
    "Findings:",
    "1. Report EXP-2026-3382 (submitter Dean Falkner) included a $640 dinner",
    "with eight external guests; guest list missing. Recurrence of a Q2 finding.",
    "2. Report EXP-2026-3410 (submitter Rosa Camacho) duplicated a hotel folio",
    "across two trips; over-claim of $412; repayment received Nov 21.",
    "3. Two reports used the corporate card for personal airline tickets",
    "(submitter initials withheld here); repayments received Dec 1.",
    "4. One manager approved their own expense (control breach);",
    "workflow fix assigned to finance systems owner Gregor Hale.",
    "Recommendations: mandatory guest disclosure above $300,",
    "duplicate-detection rule in the expense tool, self-approval block.",
    "Management responses received from the sales VP and finance director.",
    "Next review: Q2 next year.",
    "Audit team: lead auditor Sheena Patel; senior T. Okonkwo.",
    "Distribution: audit committee chair; CFO office."),
  [("RECORD_ID", "EXP-2026-3382", True, "expense_reference"),
   ("PERSON", "Dean Falkner", True, "employee_identity"),
   ("RECORD_ID", "EXP-2026-3410", True, "expense_reference"),
   ("PERSON", "Rosa Camacho", True, "employee_identity"),
   ("PERSON", "Gregor Hale", True, "private_person"),
   ("PERSON", "Sheena Patel", True, "private_person"),
   ("PERSON", "Okonkwo", True, "employee_identity")],
  [("FINANCIAL", "$640", True, "expense_amount"),
   ("FINANCIAL", "$412", True, "over_claim_amount")])

D("case_078", "en", "very_long", "nonprofit",
  P("Scholarship program administration notes (example)",
    "Program: City STEM Bursary, round 2026-B. Applications received: 214.",
    "Shortlist meeting held Dec 9; committee of five.",
    "Awardee 1: student A., 17, will study mechanical engineering;",
    "family income verified below threshold; award $6,000.",
    "Awardee 2: student B., 18, first in family to attend university;",
    "award $6,000; mentor assigned (volunteer engineer R. Castellanos).",
    "Awardee 3: student C., 16, early-admission physics candidate;",
    "award $4,000 plus laptop grant.",
    "Full names and contact details are held in the secure registry;",
    "this narrative copy intentionally omits them.",
    "Disbursement: two installments, Feb and Sep, to verified accounts.",
    "Coordinator: office manager Bethany Cole, (555) 010-2288 office line.",
    "Data retention: applicant data for non-awardees purged after 13 months.",
    "Board signer: program director H. Lindqvist."),
  [("RECORD_ID", "2026-B", True, "program_reference"),
   ("PERSON", "R. Castellanos", True, "private_person"),
   ("PERSON", "Bethany Cole", True, "private_person"),
   ("PHONE", "(555) 010-2288", False, "business_hotline"),
   ("PERSON", "Lindqvist", True, "private_person")],
  [("FINANCIAL", "award $6,000", True, "award_amount")])

D("case_079", "en", "long", "vehicle_service",
  P("Dealership service order (example)",
    "Customer: Dana Whitlock. Vehicle: 2022 EV hatchback, VIN tail 8821.",
    "Mileage: 24,180. Complaint: reduced range after software update 12.4.",
    "Diagnostics: battery health 94%; no cell faults; range calculator reset.",
    "Actions: recalibrated BMS, test drive 30 km, range restored to spec.",
    "Advisories: rear wiper blade worn; left alloy curb rash.",
    "Costs: diagnostics $180 (waived - warranty); loaner car provided.",
    "Next visit: annual service in June; recall notice 26E11 pending parts.",
    "Service advisor: Roman Vick. Courtesy shuttle driver: Marv."),
  [("PERSON", "Dana Whitlock", True, "private_person"),
   ("RECORD_ID", "8821", True, "vin_tail"),
   ("PERSON", "Roman Vick", True, "employee_identity")])

D("case_080", "en", "long", "quasi_veteran",
  P("Veterans support group intake summary (example)",
    "Participant: retired staff sergeant, age 58, served 1990-2014.",
    "Referral: VA social worker; group meets Thursdays.",
    "Presenting concerns: sleep disruption, hypervigilance in crowds.",
    "Medication review with consent: prazosin at night.",
    "Housing: stable apartment; commuting by bus 40 minutes.",
    "Family context: divorced; two adult children; one grandchild.",
    "Group facilitator: licensed counselor A. Brennan.",
    "Peer contact (with both parties' consent): retired medic J. Sandoval.",
    "Crisis plan on file; 24/7 line numbers shared at intake.",
    "Note: identifying details stored in the clinical system only."),
  [("MEDICAL", "prazosin at night", True, "medication"),
   ("MEDICAL", "sleep disruption, hypervigilance", True, "presenting_concerns")],
  risk_group="quasi_identifier",
  notes="age+service period+role+family context combination")


# ---------------------------------------------------------------------------
# case_081-090: Mixed / structured (JSON, YAML, SQL, Python, Shell, URI, logs,
# email, config, Markdown)
# ---------------------------------------------------------------------------

# Synthetic credential values are assembled from inert fragments so that no
# complete scanner-triggering literal appears in this generator or fixture.

_ALIYUN_TAIL = "SAMPLE" + "0KEY00TAIL"
_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"  # AWS-documented example literal
_ALIYUN_AK = "LTAI" + "SAMPLE00ONLY00FAKE00AKID"
_DEPLOY_TOKEN = "github_pat_" + "A1b2C3d4E5f6" + "g7H8i9J0k1L2" + "_" + "z" * 59
_GH_FINE_PAT = "github_pat_" + "00aabbccddeeff00112233" + "_" + "S" * 59

D("case_081", "mixed", "long", "json_config",
  P("{",
    '  "app": "billing-worker",',
    '  "database": {',
    '    "url": "postgres://billing_ro:R0_Billing#2026@db.int.example.net:5432/billing",',
    '    "pool": 8',
    '  },',
    '  "aws": {',
    '    "access_key_id": "' + _AWS_KEY + '",',
    '    "region": "us-west-2"',
    '  },',
    '  "oncall": "sre-billing@example-corp.net",',
    '  "dashboard": "https://grafana.int.example.net/d/billing"',
    "}"),
  [("DATABASE_URI", "postgres://billing_ro:R0_Billing#2026@db.int.example.net:5432/billing", True, "database_credential"),
   ("SECRET", _AWS_KEY, True, "cloud_credential"),
   ("EMAIL", "sre-billing@example-corp.net", False, "corporate_role_mailbox"),
   ("PRIVATE_URL", "https://grafana.int.example.net/d/billing", True, "internal_dashboard")])

D("case_082", "mixed", "long", "yaml_deploy",
  P("staging:",
    "  replicas: 2",
    "  env:",
    "    ALIYUN_AK_ID: \"" + "LTAI" + _ALIYUN_TAIL + "\"",
    "    REDIS_URL: \"redis://:R3d1sStaging!@cache-stg.example.net:6379/3\"",
    "    SMS_SIGN: \"ExampleNotify\"",
    "  alert_receivers:",
    "    - 13800009001",
    "    - 13800009002"),
  [("SECRET", "LTAI" + _ALIYUN_TAIL, True, "cloud_credential"),
   ("DATABASE_URI", "redis://:R3d1sStaging!@cache-stg.example.net:6379/3", True, "database_credential"),
   ("CN_PHONE_NUMBER", "13800009001", True, "private_contact"),
   ("CN_PHONE_NUMBER", "13800009002", True, "private_contact")])

D("case_083", "mixed", "long", "sql_dump",
  P("-- example dump (synthetic rows)",
    "INSERT INTO users (id, name, phone, email, created_at) VALUES",
    "  (1, 'Xu Ruo', '13800001001', 'xuruo@sample-mail.cn', '2026-01-04'),",
    "  (2, 'Fang Qi', '13800001002', 'fangqi@sample-mail.cn', '2026-01-11');",
    "INSERT INTO api_tokens (user_id, token) VALUES (2, 'sk-sample-0000000000000000token');"),
  [("CN_NAME", "Xu Ruo", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800001001", True, "private_contact"),
   ("EMAIL", "xuruo@sample-mail.cn", True, "private_contact"),
   ("CN_NAME", "Fang Qi", True, "private_person"),
   ("CN_PHONE_NUMBER", "13800001002", True, "private_contact"),
   ("EMAIL", "fangqi@sample-mail.cn", True, "private_contact"),
   ("SECRET", "sk-sample-0000000000000000token", True, "api_credential")])

D("case_084", "mixed", "medium", "python_script",
  P("# sync_users.py - example script",
    "ALIYUN_AK_ID = \"" + _ALIYUN_AK + "\"",
    "",
    "def fetch(db_uri):",
    "    # mysql://sync_user:Sync#2026@10.9.9.40:3306/users",
    "    return query(db_uri)",
    "",
    "DEPLOY_TOKEN = \"" + _DEPLOY_TOKEN + "\""),
  [("SECRET", _ALIYUN_AK, True, "cloud_credential"),
   ("DATABASE_URI", "mysql://sync_user:Sync#2026@10.9.9.40:3306/users", True, "database_credential"),
   ("SECRET", _DEPLOY_TOKEN, True, "api_credential")])

D("case_085", "mixed", "medium", "shell_script",
  P("#!/bin/bash",
    "# deploy-guard.sh (example)",
    "GH_TOKEN=\"" + _GH_FINE_PAT + "\"",
    "curl -s -H \"Authorization: token $GH_TOKEN\" https://api.github.example.internal/repos/acme/api > /dev/null",
    "MONGO=\"mongodb://ops_writer:Wr1te0nly%23@mongo-stg.example.net:27017/ops\"",
    "echo \"deploy done\""),
  [("SECRET", _GH_FINE_PAT, True, "api_credential"),
   ("PRIVATE_URL", "https://api.github.example.internal/repos/acme/api", True, "internal_endpoint"),
   ("DATABASE_URI", "mongodb://ops_writer:Wr1te0nly%23@mongo-stg.example.net:27017/ops", True, "database_credential")])

D("case_086", "mixed", "long", "logs",
  P("2026-12-19T08:21:03Z INFO  auth session=88213 user=m.nakamura ip=10.30.4.88 action=login",
    "2026-12-19T08:21:07Z WARN  rate-limit user=m.nakamura attempts=5",
    "2026-12-19T08:22:10Z INFO  mailer to=h.muller72@example-mail.de template=invoice",
    "2026-12-19T08:23:44Z ERROR backup target 10.30.4.91 unreachable (timeout 30s)",
    "2026-12-19T08:25:02Z INFO  auth user=s.okafor ip=10.30.7.12 action=login mfa=ok",
    "2026-12-19T08:26:31Z WARN  disk usage /var/lib/db 87%"),
  [("RECORD_ID", "88213", True, "session_reference"),
   ("USERNAME", "m.nakamura", True, "account_handle"),
   ("IP_ADDRESS", "10.30.4.88", True, "internal_infrastructure"),
   ("EMAIL", "h.muller72@example-mail.de", True, "private_contact"),
   ("IP_ADDRESS", "10.30.4.91", True, "internal_infrastructure"),
   ("USERNAME", "s.okafor", True, "account_handle"),
   ("IP_ADDRESS", "10.30.7.12", True, "internal_infrastructure")])

D("case_087", "mixed", "long", "email_full",
  P("Received: from mx3.example-mail.net (10.2.14.7)",
    "  by imap.personal-mail.cn with ESMTPS id 88213999",
    "From: 严冬 <yandong.personal@example-mail.cn>",
    "To: hr-team@company-hr.example.com",
    "Subject: 请假与医院证明",
    "",
    "您好，",
    "我因急性阑尾炎于 12 月 17 日在市一医院手术，住院 4 天，",
    "预计 12 月 24 日复工，假条随后补交。手续费 my id 是 330106199001015513，",
    "报销时需要提供给财务。手机 13800007001。",
    "",
    "严冬"),
  [("IP_ADDRESS", "10.2.14.7", True, "internal_infrastructure"),
   ("EMAIL", "yandong.personal@example-mail.cn", True, "private_contact"),
   ("EMAIL", "hr-team@company-hr.example.com", False, "corporate_role_mailbox"),
   ("CN_NAME", "严冬", True, "private_person")],
  [("MEDICAL", "急性阑尾炎", True, "diagnosis"),
   ("MEDICAL", "市一医院手术", True, "treatment")],
  notes="NOTE: the ID number string is written in prose; gold relies on numeric rule",
  )

D("case_088", "mixed", "very_long", "nginx_config",
  P("# /etc/nginx/conf.d/portal.conf (example)",
    "upstream portal_backend {",
    "    server 10.40.1.11:8443;",
    "    server 10.40.1.12:8443 backup;",
    "}",
    "",
    "server {",
    "    listen 443 ssl;",
    "    server_name portal.example-corp.net;",
    "",
    "    ssl_certificate     /etc/pki/portal/fullchain.pem;",
    "    ssl_certificate_key /etc/pki/portal/privkey.pem;",
    "",
    "    location /api/ {",
    "        proxy_pass https://portal_backend;",
    "        proxy_set_header X-Real-IP $remote_addr;",
    "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;",
    "    }",
    "",
    "    location /health {",
    "        return 200 'ok';",
    "    }",
    "}",
    "",
    "# legacy redirect for the old intranet",
    "server {",
    "    listen 80;",
    "    server_name portal-old.example-corp.net;",
    "    return 301 https://portal.example-corp.net$request_uri;",
    "}",
    "",
    "# ops notes (do not commit)",
    "# DB jump host: 10.40.1.30 (bastion), user ops-jump, key id RSA-8821",
    "# On-call roster page: https://wiki.int.example-corp.net/oncall",
    "# Escalation email: incident-bridge@example-corp.net"),
  [("IP_ADDRESS", "10.40.1.11", True, "internal_infrastructure"),
   ("IP_ADDRESS", "10.40.1.12", True, "internal_infrastructure"),
   ("IP_ADDRESS", "10.40.1.30", True, "internal_infrastructure"),
   ("PRIVATE_URL", "https://wiki.int.example-corp.net/oncall", True, "internal_wiki"),
   ("EMAIL", "incident-bridge@example-corp.net", False, "corporate_role_mailbox")],
  notes="length padded via repeated config blocks to cross 3000 chars")

D("case_089", "mixed", "medium", "docker_compose",
  P("version: \"3.8\"",
    "services:",
    "  app:",
    "    image: registry.example-corp.net/team/app:2026.12",
    "    environment:",
    "      - DATABASE_URI=mysql://app_svc:App#Svc2026@mysql-main.example.net:3306/appdb",
    "      - ACCESS_KEY=SYNTH_ACCESS_KEY_089_SAMPLE",
    "      - SECRET=SYNTH_SECRET_KEY_089_SAMPLE001",
    "    ports:",
    "      - \"8443:8443\"",
    "  worker:",
    "    image: registry.example-corp.net/team/worker:2026.12",
    "    command: [\"--queue\", \"billing\"]"),
  [("PRIVATE_URL", "registry.example-corp.net/team/app:2026.12", True, "internal_registry"),
   ("DATABASE_URI", "mysql://app_svc:App#Svc2026@mysql-main.example.net:3306/appdb", True, "database_credential"),
   ("SECRET", "SYNTH_ACCESS_KEY_089_SAMPLE", True, "cloud_credential"),
   ("SECRET", "SYNTH_SECRET_KEY_089_SAMPLE001", True, "cloud_credential")])

D("case_090", "mixed", "very_long", "markdown_meeting",
  P("# Q4 platform review (example minutes)",
    "",
    "## Attendees",
    "",
    "- Platform lead: Ash Nair",
    "- SRE: 张伟 (Wei Zhang)",
    "- Security: Fatima Noor",
    "- PM: Julia Stein",
    "",
    "## Incidents this quarter",
    "",
    "Three SEV-2 incidents, one SEV-1. The SEV-1 (Dec 3, 41 minutes) was caused",
    "by a bad migration; roll-forward fix shipped at 15:42.",
    "",
    "## Access review findings",
    "",
    "Two stale deploy keys found. Rotated during the meeting:",
    "",
    "    key id: SHA256:0000000000000000000000000000000000000000000 (old, revoked)",
    "    key id: SHA256:1111111111111111111111111111111111111111111 (old, revoked)",
    "",
    "## Credentials hygiene",
    "",
    "The scan found one hardcoded token in a test fixture (reported to owner),",
    "and one AWS-style example key that must move to the vault:",
    "",
    "    " + _AWS_KEY,
    "",
    "## Roadmap",
    "",
    "- Jan: multi-region failover drill",
    "- Feb: zero-downtime migrations",
    "- Mar: SSO for all internal tools",
    "",
    "## Contacts",
    "",
    "Security escalations: security-oncall@example-corp.net",
    "After-hours bridge: +86 571 8888 0000 (24/7)",
    "",
    "## Appendix: environment matrix",
    "",
    "| env | db host | replicas | backup schedule |",
    "|-----|---------|----------|-----------------|",
    "| prod | 10.50.1.10 | 3 | daily 02:00 |",
    "| stg | 10.50.2.10 | 1 | daily 03:00 |",
    "| dev | 10.50.3.10 | 1 | weekly |",
    "",
    "(Minutes continue with detailed migration notes for the remaining",
    "quarter; synthetic rows repeat below for length coverage.)",
    "",
    "The migration notes repeat the same backup windows for stg (daily 03:00,",
    "host 10.50.2.10) and dev (weekly, host 10.50.3.10), and reiterate that",
    "the prod host is 10.50.1.10 with three replicas and daily 02:00 backups.",
    "The Q4 platform review was led by Ash Nair with 张伟 (Wei Zhang) covering",
    "SRE topics, Fatima Noor covering security posture, and Julia Stein taking",
    "product minutes. Escalations go to security-oncall@example-corp.net and",
    "the after-hours bridge at +86 571 8888 0000."),
  [("PERSON", "Ash Nair", True, "private_person"),
   ("CN_NAME", "Wei Zhang", True, "private_person"),
   ("CN_NAME", "张伟", True, "private_person"),
   ("PERSON", "Fatima Noor", True, "private_person"),
   ("PERSON", "Julia Stein", True, "private_person"),
   ("SECRET", _AWS_KEY, True, "cloud_credential"),
   ("EMAIL", "security-oncall@example-corp.net", False, "corporate_role_mailbox"),
   ("PHONE", "+86 571 8888 0000", False, "business_hotline"),
   ("IP_ADDRESS", "10.50.1.10", True, "internal_infrastructure"),
   ("IP_ADDRESS", "10.50.2.10", True, "internal_infrastructure"),
   ("IP_ADDRESS", "10.50.3.10", True, "internal_infrastructure")])


# ---------------------------------------------------------------------------
# case_091-100: PII-free / public / ambiguous
# ---------------------------------------------------------------------------

D("case_091", "zh", "medium", "public_news",
  P("某科技公司发布年度可持续报告",
    "该报告显示，公司全年清洁能源使用比例提升至 42%，数据中心平均 PUE 降至 1.18。",
    "公司表示，下一阶段将把液冷方案推广到更多自建机房，并持续开放节能专利。",
    "多语言版本：العربية / Español / Français。"),
  [],
  notes="pure public corporate news, no PII")

D("case_092", "en", "medium", "public_docs",
  P("Product changelog v2.4.0",
    "- Add bulk export to CSV for enterprise plans.",
    "- Fix a timezone bug in scheduled reports.",
    "- Improve login performance by caching session lookups.",
    "- Deprecate the legacy webhook format; migration guide available."),
  [],
  notes="public changelog, no PII")

D("case_093", "zh", "short", "weather",
  "明天白天多云，夜间转小雨，气温 12 到 17 度，东北风三级。",
  [],
  notes="weather bulletin, no PII")

D("case_094", "en", "short", "cooking",
  "Sift 250g flour with a pinch of salt, whisk two eggs, and fold into the batter.",
  [],
  notes="recipe step, no PII")

D("case_095", "zh", "medium", "ambiguous_hotline",
  "全国客服热线 400-880-1666，服务时间每日 8:00-22:00。",
  [("PHONE", "400-880-1666", False, "public_hotline")],
  notes="public hotline: detected, must NOT be redacted")

D("case_096", "en", "medium", "ambiguous_official",
  "Official inquiries: press.office@gov-agency.example.org; hotline 555-0100.",
  [("EMAIL", "press.office@gov-agency.example.org", False, "public_official_mailbox"),
   ("PHONE", "555-0100", False, "public_hotline")],
  notes="official public contacts: detected, must NOT be redacted")

D("case_097", "zh", "long", "regulation_text",
  P("某条例（节选，示例文本）",
    "第十二条 处理个人信息应当具有明确、合理的目的，并应当与处理目的直接相关，",
    "采取对个人权益影响最小的方式。",
    "第十三条 敏感个人信息是一旦泄露或者非法使用，容易导致自然人的人格尊严受到侵害",
    "或者人身、财产安全受到危害的个人信息，包括生物识别、宗教信仰、特定身份、医疗健康、",
    "金融账户、行踪轨迹等信息。"),
  [],
  notes="legal text quoting category names only, no PII instances")

D("case_098", "en", "long", "engineering_docs",
  P("How backpressure works (engineering blog draft)",
    "When a consumer cannot keep up with a producer, a naive queue grows without bound.",
    "Backpressure propagates demand signals upstream so the producer slows down.",
    "Reactive libraries implement this via request(n) signaling.",
    "In our pipeline the broker caps in-flight messages per partition,",
    "and consumers acknowledge in batches of 64.",
    "When latency budget is exceeded, the producer halves its batch size,",
    "recovers gradually, and surfaces a degraded-mode metric."),
  [],
  notes="technical prose, no PII")

D("case_099", "en", "short", "semantic_negative",
  "Sam mentioned going to the hospital yesterday but did not say why.",
  [],
  notes="SEMANTIC NEGATIVE: hospital mention without condition; no inference allowed",
  semantic=[("MEDICAL", "going to the hospital", False, "hospital_mention_without_condition")])

D("case_100", "zh", "medium", "semantic_negative",
  "小李昨天去了医院，不过没有说明具体原因。",
  [],
  notes="SEMANTIC NEGATIVE (goal 1.5): must NOT infer illness",
  semantic=[("MEDICAL", "去了医院", False, "hospital_mention_without_condition")])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Length extensions for very_long documents (must exceed 3000 chars each)
# ---------------------------------------------------------------------------

_ext("case_039")([
    "七、二审情况（模拟）",
    "被告不服一审判决提起上诉，主张已另行支付现金 6,000 元并提交了无签收凭证的手写收条。",
    "二审期间，经法院主持调解，双方自愿达成调解协议：被告于十五日内一次性支付 9,000 元；",
    "原告放弃其他请求；一审诉讼费由被告负担，二审诉讼费减半收取。",
    "调解书经双方签收后发生法律效力。",
    "八、法官后语（模拟教学用）",
    "本案争议焦点在于劳务关系解除的合法性审查与工资支付事实的举证责任分配。",
    "用人单位单方解除时，应就解除所依据的规章制度、事实证据及程序合规承担举证责任；",
    "劳动者主张欠付工资的，应就劳动事实与欠付数额提供初步证据。",
    "用人单位以现金方式支付工资的，应当保留经劳动者签收的凭证，否则将承担不利后果。",
    "九、庭前会议记录（模拟）",
    "庭前会议于 12 月 5 日 09:30 在第 5 法庭召开，双方委托诉讼代理人到庭。",
    "会议明确争议焦点三项：欠付工资数额、解除合法性、年假折算。",
    "原告补充提交了银行流水补充页与微信催讨记录公证材料；被告申请延长举证期限七日，",
    "合议庭评议后予以准许。",
])

_ext("case_040")([
    "七、评估与定价方法（备忘录附录）",
    "本次交易拟采用收益法与市场法交叉验证：收益法以乙方未来五年现金流预测为基础，",
    "折现率参考可比公司 WACC 区间 10.5%-12.3%；市场法选取三起可比交易，EV/EBITDA 倍数区间",
    "为 9.8x-13.5x。评估机构将于 1 月 8 日前出具正式报告。",
    "八、过渡期安排",
    "自签署日起至交割日，乙方日常经营按既有预算执行；单笔超过 500 万元的支出须书面通知甲方；",
    "核心技术人员名单（附录三，共 27 人）适用留任奖金安排。",
    "九、陈述与保证要点",
    "乙方就土地使用权、环评批复、知识产权权属及未决诉讼作出陈述与保证；",
    "甲方就支付能力与内部授权作出陈述与保证。",
    "十、争议解决",
    "因本备忘录引起的争议适用仲裁，仲裁机构为某国际经济贸易仲裁委员会，仲裁地 上海。",
    "十一、附件清单",
    "附件一 股东名册；附件二 专利清单；附件三 核心技术人员名单；附件四 重大合同汇总；",
    "附件五 过渡期预算表；附件六 数据室索引。",
    "项目组每周一 10:00 例会；数据室新增资料将于 24 小时内同步至索引。",
])

_ext("case_044")([
    "六、取证与复盘补充",
    "终端镜像哈希值已记录并双人复核；导出行为共 47 次，时间跨度 11 月 28 日至 12 月 14 日，",
    "其中 43 次命中同一条 SQL 模板。导出数据经脱敏审计确认未进入公网邮箱，仅落地至本机加密分区。",
    "七、监管沟通口径（节选）",
    "公司将于 72 小时内完成初步通报，说明事件范围、影响评估与整改计划；",
    "后续按监管要求补充取证结论与客户告知方案。",
    "八、客户告知方案",
    "首批告知对象为确认存在导出的客户名单，通知渠道为短信与站内信，",
    "内容包含事件概述、影响字段、已采取措施与咨询渠道；不包含任何可被用于钓鱼的链接。",
    "九、内部培训安排",
    "数据分级、最小权限与导出审批三项课程合并为 90 分钟必修课，",
    "分三批实施，覆盖全部数据岗位；考核不合格者暂停数据访问权限。",
    "十、附录：处置人员名单",
    "指挥：信息安全部经理；技术处置：SOC 值班两组；法务接口：诉讼与合规岗；",
    "公关接口：品牌与公共事务部；行政支持：总务部。",
])

_ext("case_045")([
    "八、资格预审结果说明",
    "三家企业均通过资格预审；其中一家因业绩证明材料补正一次，评审委员会同意延后两日提交。",
    "九、技术评审要点",
    "售检票系统需支持刷码、刷脸与实体卡三种介质；关键指标包括通道通行速度不低于 35 人/分钟、",
    "离线模式可存储 30 万笔交易、断网恢复后自动补传。",
    "十、商务评审要点",
    "报价不得超过最高限价 1.86 亿元；质保期三年，备件响应时间市内 4 小时。",
    "十一、廉洁与保密",
    "评标专家名单在开标前严格保密；投标人不得以任何形式打听或接近评标委员会成员；",
    "违反者取消投标资格并记入不良行为记录。",
    "十二、后续安排",
    "中标公示三个工作日；合同谈判预计两周；计划 3 月 1 日进场，",
    "首列车载设备到货时间为 5 月 15 日。",
    "十三、项目沟通机制",
    "每周三 14:00 视频例会；重要事项以书面联系单为准；联系单编号规则 METRO6-AFC-LXD-XXX。",
])

_ext("case_076")([
    "Additional scene details (training continuation):",
    "Responding units secured the perimeter and canvassed the block for cameras.",
    "Two neighboring residences reported a silver sedan idling near the alley",
    "between 21:40 and 22:00; the partial plate on file ends in 7K.",
    "Detective unit scheduled a follow-up interview with the reporting neighbor",
    "and requested the block association's private camera footage.",
    "The resident provided a purchase list for the missing laptop,",
    "including an accessory serial that may aid recovery.",
    "Patrol recommendations for the block: improved rear-gate lighting,",
    "trimming of the overgrown hedge along the alley, and a neighborhood watch",
    "rotation covering the 21:00-23:00 window.",
    "All identifiers in this scenario are synthetic; any resemblance to real",
    "persons or case numbers is unintended.",
])

_ext("case_077")([
    "Appendix A - sampling rationale:",
    "The sample targeted submitters with Q2 findings, trip segments above $5,000,",
    "and first-time international itineraries. Two submitters accounted for",
    "eleven reports and were sampled at 100%.",
    "Appendix B - control walkthrough:",
    "The expense tool enforces receipt attachment above $75 and flags",
    "same-merchant duplicates within 30 days. The duplicate-detection gap",
    "involved trips 44 days apart, outside the current window.",
    "Appendix C - prior year comparison:",
    "Q3 last year reported five findings across three divisions; the sales",
    "division accounted for four. Guest-disclosure compliance improved from",
    "71% to 88% after the September policy refresh.",
    "Appendix D - definitions:",
    "High-value guest dinner: any single meal above $300 with external guests.",
    "Duplicate folio: identical merchant, amount, and folio number across trips.",
    "Self-approval: any report where the approver and submitter IDs match.",
])

_ext("case_078")([
    "Appendix - review rubric (summary):",
    "Academic readiness 40%, financial need 30%, community contribution 20%,",
    "essay quality 10%. Two committee members scored independently; divergences",
    "above ten points triggered discussion.",
    "Waitlist notes:",
    "Three alternates were ranked and will be contacted only if an awardee declines.",
    "Alternate files mirror awardee files in the secure registry with a",
    "distinct retention clock.",
    "Mentor program onboarding:",
    "Volunteer mentors complete a background check and a two-hour training",
    "covering boundaries, meeting logistics, and escalation paths.",
    "Mentor-mentee matches are reviewed at the six-week mark.",
    "Communications calendar:",
    "Award letters in January; disbursement confirmations in February and",
    "September; alumni survey in June. All correspondence uses office templates.",
])

_ext("case_088")([
    "",
    "# additional vhost: metrics portal (example)",
    "server {",
    "    listen 443 ssl;",
    "    server_name metrics.example-corp.net;",
    "    location / {",
    "        proxy_pass http://metrics_backend;",
    "        auth_request /_auth_verify;",
    "    }",
    "    location /_auth_verify {",
    "        proxy_pass http://authsidecar.internal/verify;",
    "        proxy_pass_request_body off;",
    "    }",
    "}",
    "",
    "# additional vhost: static archive",
    "server {",
    "    listen 443 ssl;",
    "    server_name archive.example-corp.net;",
    "    root /srv/archive;",
    "    autoindex off;",
    "    location ~* \\.(key|pem|env)$ {",
    "        deny all;",
    "    }",
    "}",
    "",
    "# rate limiting notes for the api gateway",
    "# limit_req_zone $binary_remote_addr zone=api:10m rate=20r/s;",
    "# burst=40 nodelay applies to the /api/ location above",
    "",
    "# log shipping",
    "# access logs ship to the collector at 10.40.1.40:514 via syslog,",
    "# retention 30 days hot, 180 days cold",
])

_ext("case_090")([
    "",
    "## Appendix B: capacity planning",
    "",
    "The queue was resized from 12 to 18 partitions to absorb the seasonal",
    "traffic; consumer group lag stayed under 4 seconds at the 95th percentile.",
    "The db host for prod (10.50.1.10) now runs read replicas in two zones.",
    "",
    "## Appendix C: runbook updates",
    "",
    "Failover drill steps were updated; the on-call bridge number",
    "+86 571 8888 0000 remains the single after-hours path, and",
    "security-oncall@example-corp.net must be cc'd on all SEV-1 timelines.",
    "张伟 (Wei Zhang) owns the drill; Ash Nair signs off the report;",
    "Fatima Noor reviews the access-log retention changes; Julia Stein",
    "communicates the customer-visible maintenance window.",
    "",
    "## Appendix D: follow-ups",
    "",
    "1. Vault migration for the remaining example credential - owner: SRE.",
    "2. Deploy key inventory - owner: Security.",
    "3. Backup restore test for 10.50.1.10 - owner: DBA.",
    "4. Quarterly access review cadence - owner: Compliance.",
])


# ---------------------------------------------------------------------------
# Second-round length extensions (ensure >3000 chars for very_long docs)
# ---------------------------------------------------------------------------

_ext("case_039")([
    "十、相关法条摘引（模拟教学用）",
    "关于工资支付：工资应当以货币形式按月支付给劳动者本人，不得克扣或者无故拖欠。",
    "关于解除程序：用人单位单方解除的，应当事先将理由通知工会；研究决定解除的，",
    "应当事先将理由通知工会，工会认为不适当的，有权提出意见。",
    "关于举证：发生劳动争议，当事人对自己提出的主张，有责任提供证据；",
    "与争议事项有关的证据属于用人单位掌握管理的，用人单位应当提供。",
    "十一、教学讨论题",
    "1. 本案中冷库温度记录造假与解除协议之间的因果关系如何认定？",
    "2. 手写收条在无签收凭证情形下的证明力如何评价？",
    "3. 劳务关系与劳动关系的区分对本案程序选择有何影响？",
    "4. 若原告同时主张未签书面合同的双倍工资，举证结构将如何变化？",
    "5. 调解协议生效后一方拒不履行的救济路径是什么？",
    "十二、附表（模拟数据）",
    "工资明细表：8 月基本工资 12,000 元，出勤 21 天；9 月基本工资 12,000 元，出勤 19 天；",
    "加班补贴合计 1,860 元；代扣代缴项目合计 2,340 元；实发口径以银行流水为准。",
    "年假折算：应休 5 天，已休 2 天，折算 3 天，日基数为月工资除以 21.75。",
    "（本案全部内容为模拟教学材料，不构成任何法律意见。）",
])

_ext("case_040")([
    "十二、交割前置条件清单",
    "1. 反垄断申报获得批准或不实施进一步审查的通知；",
    "2. 乙方董事会及股东会批准交易文件；",
    "3. 银行债权人出具贷款延续或清偿确认函；",
    "4. 核心技术人员留任协议签署率达到 80%；",
    "5. 无重大不利变化条款核查通过。",
    "十三、数据与系统整合规划（摘要）",
    "双方 ERP 系统将于交割后 90 日内完成主数据对齐；客户与供应商编码采用映射表过渡；",
    "乙方生产基地的 MES 系统保留本地部署，数据每日同步至集团数据平台；",
    "统一身份认证接入集团 SSO，历史口令强制轮换；",
    "涉密图纸与工艺文件迁移至集团文档系统并按密级重新授权。",
    "十四、关键时间线（更新）",
    "1 月 20 日前完成环境、社会与治理尽调补充；2 月 10 日前签署正式协议；",
    "2 月 28 日前完成反垄断申报；预计 5 月 31 日交割。",
    "十五、沟通纪律",
    "项目相关沟通一律使用项目邮箱与项目群，禁止使用个人社交软件传输交易文件；",
    "本项目对外口径由董事会办公室统一发布。",
])

_ext("case_044")([
    "十一、技术复盘（节选）",
    "导出行为使用的报表系统存在两处缺陷：其一，导出接口未对接细粒度行级权限；",
    "其二，审计日志只记录到表级，未记录到行级过滤条件，导致影响面评估耗时较长。",
    "整改版本将在下季度上线，包含行级水印与导出二次审批。",
    "十二、客户名单核对过程",
    "安全团队使用受控副本在隔离环境完成名单比对，全程录屏；",
    "比对结果经双人复核后封存，原始镜像保持只读。",
    "十三、时间线复盘结论",
    "从告警到冻结账号用时 7 分钟，属于达标区间；从冻结到完成影响面初评用时 40 分钟，",
    "超出自标 25 分钟，主要原因是行级统计能力缺失，已列入整改项。",
    "十四、后续审计安排",
    "内审部将于下月对整改项进行独立验证；验证材料包括策略配置快照、",
    "审批流截图与抽样导出记录；验证结论提交信息安全委员会。",
])

_ext("case_045")([
    "十四、评标细则补充",
    "技术方案评分细目：系统架构 15 分、设备性能 15 分、实施组织 10 分、",
    "售后服务 10 分、培训方案 5 分、本地化支持 5 分。",
    "商务评分细目：企业业绩 10 分、财务状况 5 分、信用记录 5 分。",
    "价格分以有效报价的算术平均值为基准价，每高于基准价 1% 扣 1 分，每低于 1% 扣 0.5 分。",
    "十五、样机测试安排",
    "通过符合性审查的投标人须在指定站点完成 1,000 次模拟过闸测试，",
    "测试数据由第三方检测机构采集并封存，作为技术评分依据。",
    "十六、应急预案",
    "如遇不可抗力导致开标延期，另行公告；投标人应确保联系方式畅通，",
    "因联系方式失效造成的后果由投标人自行承担。",
    "十七、本项目廉政承诺",
    "所有项目参与人员签署廉政承诺书；发现围标串标线索的，移送有关机关处理。",
])

_ext("case_076")([
    "Detective follow-up (training continuation):",
    "The detective unit obtained a warrant for the alley camera footage;",
    "the provider delivered 40 minutes of clips. A person of interest was",
    "observed leaving the alley on foot at 21:52 carrying a rectangular bag.",
    "The clip was logged into evidence under the case number above.",
    "Neighborhood canvass summary: eight of eleven households answered;",
    "two reported hearing a metallic sound around 21:50,",
    "consistent with the forced rear gate.",
    "The resident submitted a replacement list for documents stored in the",
    "stolen laptop bag, including an insurance card and a vehicle registration.",
    "Prevention follow-up: the community liaison scheduled a block meeting,",
    "and the property manager agreed to install a motion light at the rear gate.",
    "All names and numbers in this scenario remain synthetic.",
])

_ext("case_077")([
    "Appendix E - interviews conducted:",
    "The team interviewed the expense tool administrator, two division",
    "controllers, and the corporate card program manager.",
    "Key control observation: approval limits are enforced by the tool,",
    "but the delegations list had not been reviewed for 14 months.",
    "Appendix F - data analytics:",
    "A full-population duplicate test flagged 61 potential pairs;",
    "manual review confirmed two genuine duplicates, both in the sample.",
    "Merchant category analysis found one card used for gambling-adjacent",
    "merchant codes; the card was cancelled and the matter referred to HR.",
    "Appendix G - timeline of audit fieldwork:",
    "Kickoff Oct 14; interim testing Nov 6-15; wrap-up Dec 2;",
    "draft report issued Dec 12; management responses received Dec 19.",
    "Appendix H - glossary of system roles:",
    "Submitter, Approver, Delegate, Auditor (read-only), and Finance Processor.",
])

_ext("case_078")([
    "Appendix - budget and funding sources:",
    "The bursary is funded by an annual gala, two corporate sponsors,",
    "and an endowment draw. The endowment draw is capped at 4% of the",
    "trailing three-year average balance.",
    "Appendix - verification standards:",
    "Household income verification accepts the two most recent tax notices",
    "or a signed attestation when tax filing is not available.",
    "Two flagged applications were referred for secondary review;",
    "both were resolved with additional documents within nine days.",
    "Appendix - equipment grants:",
    "Laptop grants follow a fixed specification; devices are enrolled in",
    "device management before handover and remain program property",
    "for the first two years.",
    "Appendix - privacy practice:",
    "Application data is stored in the secure registry with role-based",
    "access; committee members view anonymized scoring copies.",
])

_ext("case_088")([
    "",
    "# maintenance window notes (example)",
    "# The portal restarts during the monthly window at 02:00 on the first",
    "# Tuesday. Health checks gate the rollout; rollback is automatic when",
    "# two consecutive health checks fail within five minutes.",
    "",
    "server {",
    "    listen 8443 ssl;",
    "    server_name status.example-corp.net;",
    "    location / {",
    "        return 302 /status-page;",
    "    }",
    "    location /status-page {",
    "        proxy_pass http://status_backend;",
    "    }",
    "}",
    "",
    "# change log (excerpt)",
    "# 2026-11-04 added metrics portal vhost",
    "# 2026-11-22 rotated portal certificate (validity 90 days)",
    "# 2026-12-06 tuned rate limit burst from 30 to 40",
    "# 2026-12-18 added archive vhost deny rule for key material",
    "",
    "# contacts for this file",
    "# owner: platform team; reviewer: security team;",
    "# escalations: incident-bridge@example-corp.net",
    "# next review: end of January",
])

_ext("case_090")([
    "",
    "## Appendix E: migration runbook (abridged)",
    "",
    "1. Freeze deploys 30 minutes before the window.",
    "2. Snapshot the prod database on 10.50.1.10 and verify restore on stg.",
    "3. Apply schema changes with the online DDL tool; watch lock time.",
    "4. Smoke test the checkout path; verify the payments queue depth.",
    "5. Rollback criteria: error rate above 1% for five minutes.",
    "",
    "The runbook is owned by 张伟 (Wei Zhang) and reviewed by Ash Nair;",
    "Fatima Noor signs off the access changes and Julia Stein posts the",
    "customer notice. Escalations after hours use +86 571 8888 0000 and",
    "security-oncall@example-corp.net.",
    "",
    "## Appendix F: glossary",
    "",
    "SEV-1: full outage or data integrity risk; bridge and exec page required.",
    "SEV-2: partial degradation with a workaround; bridge required.",
    "Freeze: no deploys except security fixes.",
    "Vault: the internal secret store; no credentials in code or tickets.",
])

_ext("case_072")([
    "Extended notes - payments platform (continued):",
    "Reconciliation design: the monthly job reads settlement files from PSP A,",
    "matches on (txn id, amount, currency, timestamp bucket), and writes",
    "exceptions to a review queue; unmatched above 0.5% pages the on-call.",
    "Anita Deshmukh owns the month-end close checklist.",
    "Incident INC-7712 remediation plan:",
    "1. Add idempotency keys to the retry path (owner Tomasz Kowalski).",
    "2. Duplicate-write alarm at 0.1% within five minutes.",
    "3. Backfill script for the affected window with manual review.",
    "Access review: three service accounts had broader DB roles than needed;",
    "roles were scoped down and the review cadence moved to quarterly.",
    "The replica connection string above remains read-only;",
    "rotations happen via the vault and never land in tickets or chat.",
    "Escalation path: on-call → platform lead → VP Engineering (Nadia Osei).",
])



def _pad(case_id, chunks):
    EXTENSIONS.setdefault(case_id, []).append("".join(chunks))


_pad("case_039", [
    "十三、证据目录（模拟）\n",
] + [
    "证据{0:02d}：{1}，证明内容：{2}；提交方：{3}；质证意见：{4}。\n".format(idx, name, fact, side, reply)
    for idx, (name, fact, side, reply) in enumerate([
        ("劳务协议原件", "双方权利义务及解除条款约定", "被告", "真实性无异议，证明目的有异议"),
        ("银行流水（8-9月）", "工资支付与停发事实", "原告", "真实性无异议"),
        ("微信催讨记录公证书", "原告多次催讨未果", "原告", "对聊天主体身份提出异议"),
        ("考勤截图汇总", "原告出勤情况", "原告", "系单方截取，不完整"),
        ("卫生检查通报", "原告违反操作规范", "被告", "与解除合法性关联性不足"),
        ("员工手册签收单", "规章制度已公示", "被告", "签收不代表知晓具体条款"),
        ("工会通知函（缺失说明）", "解除程序合规性", "被告", "因交接遗漏未能提供"),
        ("年假台账", "年假安排与折算基数", "被告", "台账为单方制作"),
        ("手写收条", "被告主张的现金支付", "被告", "原告否认签署，申请笔迹鉴定"),
        ("调解笔录", "双方和解意愿与方案", "法院", "双方均无异议"),
    ], start=1)
] + [
    "十四、合议庭评议要点（模拟）\n",
    "评议认为：现金支付主张缺乏签收凭证且与工资发放惯例不符，不予采信；",
    "解除程序瑕疵不影响欠付工资的支付义务；年假折算基数采纳台账口径。",
    "十五、教学讨论题\n",
    "1. 冷库记录造假与解除之间的因果关系如何认定？2. 无签收凭证收条的证明力如何评价？",
    "3. 劳务关系与劳动关系对程序选择的影响？4. 调解书生效后拒不履行的救济路径？",
    "（本案全部内容为模拟教学材料。）\n",
])

_pad("case_040", [
    "十六、估值敏感性分析（备忘录附录）\n",
] + [
    "情景{0}：{1}；EBITDA 达成 {2}；倍数 {3}；对价区间 {4}-{5} 亿元。\n".format(idx, name, ebitda, mult, low, high)
    for idx, (name, ebitda, mult, low, high) in enumerate([
        ("基准", "3.2 亿元", "11.5x", "16.8", "18.6"),
        ("乐观", "3.6 亿元", "12.0x", "18.7", "20.4"),
        ("保守", "2.8 亿元", "10.5x", "14.2", "15.9"),
        ("压力", "2.4 亿元", "9.8x", "11.9", "13.6"),
    ], start=1)
] + [
    "十七、留任奖金方案（摘要）\n",
] + [
    "- 层级{0}：{1} 人；留任期 {2}；支付比例 {3}。\n".format(lvl, cnt, period, ratio)
    for lvl, cnt, period, ratio in [
        ("核心技术骨干", 12, "三年", "60/20/20"),
        ("关键岗位管理", 9, "两年", "50/50"),
        ("一般留任人员", 6, "一年", "100"),
    ]
] + [
    "十八、数据室完整性核对（抽样）\n",
] + [
    "- {0}：状态 {1}；责任人 {2}。\n".format(item, status, owner)
    for item, status, owner in [
        ("土地使用权证", "已归档", "法务"),
        ("环评批复（二期）", "待补", "行政"),
        ("发明专利证书", "已归档", "研发"),
        ("重大合同正本", "已归档", "商务"),
        ("未决诉讼清单", "已归档", "法务"),
        ("员工名册与社保记录", "待补", "人力"),
    ]
])

_pad("case_044", [
    "十五、导出行为明细（节选，模拟）\n",
] + [
    "- 11 月 {0:02d} 日 {1:02d}:{2:02d}：导出 {3} 行；范围：{4}；审批：无。\n".format(day, hour, minute, rows, scope)
    for day, hour, minute, rows, scope in [
        (28, 10, 12, 12000, "华东区当月新增"),
        (29, 15, 40, 9000, "华东区当月新增"),
        (30, 9, 55, 18000, "华东区全量"),
        (1, 11, 5, 22000, "华东+华南"),
        (2, 16, 30, 15000, "套餐变更记录"),
        (4, 10, 2, 26000, "全量客户主档"),
        (5, 14, 48, 26000, "全量客户主档"),
        (6, 17, 21, 31000, "主档+联系记录"),
        (8, 9, 33, 35000, "主档+联系记录"),
        (11, 20, 15, 40000, "全库抽样"),
        (14, 19, 58, 40000, "全库抽样"),
    ]
] + [
    "十六、责任认定（内部，模拟）\n",
    "涉事员工违反数据安全承诺书第 4 条；直属主管未复核导出审批，承担管理责任；",
    "报表系统负责人因权限模型缺陷承担技术责任。处理决定另行作出。\n",
])

_pad("case_045", [
    "十八、车站需求清单（模拟）\n",
] + [
    "- 车站 {0}：闸机 {1} 台；售票机 {2} 台；半自助 {3} 台；编码 {4}；批次 {5}。\n".format(name, gates, tvm, assist, code, batch)
    for name, gates, tvm, assist, code, batch in [
        ("火炬广场站", 24, 8, 4, "M6-001", "T1"),
        ("槐安东路站", 20, 6, 4, "M6-002", "T1"),
        ("科技中心站", 32, 10, 6, "M6-003", "T1"),
        ("滨河新城站", 16, 6, 2, "M6-004", "T2"),
        ("会展中心站", 28, 8, 4, "M6-005", "T2"),
        ("体育公园站", 18, 6, 2, "M6-006", "T2"),
        ("科创园区站", 22, 8, 4, "M6-007", "T3"),
        ("枢纽东站", 36, 12, 8, "M6-008", "T3"),
    ]
] + [
    "十九、备品备件（摘要）：扇门模块 40 套；读卡器 60 只；纸币模块 24 套；硬币模块 24 套；专用调试工具 8 套。\n",
    "二十、培训与移交：运维培训 40 学时（车站级 16 + 中心级 24）；考核合格率须达 90%；竣工资料按档案规范移交。\n",
])

_pad("case_076", [
    "Evidence log continuation (training scenario):\n",
] + [
    "- Item {0:02d}: {1}; collected {2}; sealed {3}; noted by Ofc. Marsh.\n".format(idx, item, where, seal)
    for idx, (item, where, seal) in enumerate([
        ("tool-mark cast", "rear door frame", "E-8821"),
        ("fingerprint lift A", "kitchen window", "E-8822"),
        ("fingerprint lift B", "kitchen window", "E-8823"),
        ("soil sample", "rear threshold", "E-8824"),
        ("fabric fibers", "gate latch", "E-8825"),
        ("neighbor camera clip", "block association", "E-8826"),
        ("gate hasp fragment", "rear gate", "E-8827"),
        ("shoe impression cast", "flower bed", "E-8828"),
    ], start=1)
] + [
    "Dec 11: evidence submitted to the lab; laptop serial flagged to",
    "secondary-market monitors. Dec 14: partial plate lead traced to a rental",
    "agency. Dec 18: lab confirms both lifts match each other; AFIS run pending.",
])

_pad("case_077", [
    "Appendix I - expense analytics by month (synthetic figures):\n",
] + [
    "- {0}: reports {1}, median ${2}, top claim ${3}, guest-disclosure {4}%.\n".format(month, count, median, top, rate)
    for month, count, median, top, rate in [
        ("July", 138, 412, 3880, 84),
        ("August", 129, 388, 4210, 86),
        ("September", 145, 431, 5640, 88),
    ]
] + [
    "Appendix J - trip segments:\n",
] + [
    "- {0}: segments {1}; avg lodging ${2}; avg airfare ${3}; anomalies {4}.\n".format(lane, segs, lodg, air, anom)
    for lane, segs, lodg, air, anom in [
        ("Domestic", 402, 189, 344, 5),
        ("Cross-border APAC", 47, 231, 689, 2),
        ("Cross-border EMEA", 21, 264, 912, 1),
    ]
] + [
    "Appendix K - remediation: guest-disclosure rule (G. Hale, Feb 15);",
    "self-approval block (vendor ticket, Jan 30); delegation review quarterly.",
])

_pad("case_078", [
    "Appendix - scoring distribution (synthetic):\n",
] + [
    "- Band {0}: {1} applications; average {2}; advancement {3}%.\n".format(name, cnt, avg, adv)
    for name, cnt, avg, adv in [
        ("A", 26, 87.4, 92),
        ("B", 64, 78.2, 58),
        ("C", 88, 69.5, 21),
        ("D", 36, 58.8, 6),
    ]
] + [
    "Interview logistics: two-slot choices; 92% attended the first choice;",
    "two reschedules due to exam conflicts; same-evening debriefs.",
    "Retention matrix: awardee files graduation+5y; non-awardees 13 months;",
    "scoring sheets 3 years anonymized; registry exports dual control.",
])

_pad("case_088", [
    "",
    "# microservice routes (example)",
] + [
    "server {{\n  listen 443 ssl;\n  server_name {0}.example-corp.net;\n"
    "  location / {{ proxy_pass http://{0}_backend; proxy_read_timeout 30s; }}\n}}\n".format(name)
    for name in ("billing", "identity", "search", "notify")
] + [
    "# change log: 2026-11-04 metrics vhost; 2026-11-22 cert rotation;",
    "# 2026-12-06 burst 30->40; 2026-12-18 archive deny rule.",
    "# contacts: platform team owner; security reviewer;",
    "# escalations: incident-bridge@example-corp.net; next review end of January.",
])

_pad("case_090", [
    "",
    "## Appendix E: migration runbook (abridged)",
    "",
    "1. Freeze deploys 30 minutes before the window.",
    "2. Snapshot the prod database on 10.50.1.10 and verify restore on stg.",
    "3. Apply schema changes with the online DDL tool; watch lock time.",
    "4. Smoke test the checkout path; verify the payments queue depth.",
    "5. Rollback criteria: error rate above 1% for five minutes.",
    "",
    "The runbook is owned by 张伟 (Wei Zhang) and reviewed by Ash Nair;",
    "Fatima Noor signs off access changes and Julia Stein posts the customer",
    "notice. After-hours escalations: +86 571 8888 0000 and",
    "security-oncall@example-corp.net.",
    "",
    "## Appendix F: glossary",
    "",
    "SEV-1: full outage or data integrity risk; bridge and exec page required.",
    "SEV-2: partial degradation with a workaround; bridge required.",
    "Freeze: no deploys except security fixes.",
    "Vault: the internal secret store; no credentials in code or tickets.",
])

_pad("case_072", [
    "Extended notes - payments platform (continued):",
    "Reconciliation design: the monthly job reads settlement files from PSP A,",
    "matches on (txn id, amount, currency, timestamp bucket), and writes",
    "exceptions to a review queue; unmatched above 0.5% pages the on-call.",
    "Anita Deshmukh owns the month-end close checklist.",
    "Incident INC-7712 remediation: idempotency keys on the retry path",
    "(owner Tomasz Kowalski); duplicate-write alarm at 0.1%; backfill script",
    "with manual review for the affected window.",
    "Access review: three service accounts had broader DB roles than needed;",
    "roles were scoped down and the review cadence moved to quarterly.",
    "The replica connection string above stays read-only; rotations happen",
    "via the vault and never land in tickets or chat.",
    "Escalation path: on-call -> platform lead -> VP Engineering (Nadia Osei).",
])



_pad("case_039", [
    "十六、庭审笔录节选（模拟）\n",
] + [
    "审判长：{0}。原告代理人：{1}。被告代理人：{2}。\n".format(q, a, b)
    for q, a, b in [
        ("请原告明确欠付工资的计算期间与依据", "2026年8月全月及9月1日至30日，依据考勤与合同约定月薪", "对9月出勤天数有异议"),
        ("请被告说明解除通知的送达方式", "书面通知于9月30日当面送达并留存副本", "原告否认收到，称仅有口头告知"),
        ("冷库温度记录造假的调查过程", "由店长复核并报区域运营，未启动正式调查程序", "程序瑕疵不影响事实认定"),
        ("年假安排的审批记录", "台账显示2026年已休2天", "原告主张另有口头批准未记载"),
        ("现金支付的收条形成过程", "被告称由店长代写原告签名", "原告申请笔迹鉴定并已预交费用"),
    ]
] + [
    "十七、鉴定意见（模拟）\n",
    "司法鉴定机构出具意见：收条上签名与原告样本签名笔迹特征存在明显差异，",
    "倾向认定非同一人书写。被告对鉴定意见未申请重新鉴定。\n",
    "（本段为模拟教学材料。）\n",
])

_pad("case_072", [
    "Extended notes - observability and tooling:",
    "Dashboards: golden signals per pod, queue lag per partition, and",
    "per-tenant error budgets; the month-end view adds settlement match rate.",
    "Alerts: settlement match rate below 99.5% pages payments-on-call;",
    "replica lag above 30 seconds warns the platform channel.",
    "Runbook drills: failover exercise runs quarterly in staging with a",
    "synthetic tenant; results are filed to the compliance workspace.",
    "Data retention: raw webhook payloads 30 days hot, 1 year cold,",
    "with PII fields masked in the cold copies.",
    "Vendor contacts (bench): PSP A integration engineer and the vault",
    "product owner are listed in the internal directory; do not email",
    "credentials or connection strings to anyone, including vendors.",
])

_pad("case_040", [
    "十九、整合风险管理（备忘录附录）\n",
] + [
    "- 风险{0}：{1}；等级 {2}；缓解措施：{3}；责任人：{4}。\n".format(idx, name, level, mitigation, owner)
    for idx, (name, level, mitigation, owner) in enumerate([
        ("核心技术人员流失", "高", "留任奖金+竞业补充协议", "人力"),
        ("客户认证延期", "中", "双团队驻厂推进", "项目组"),
        ("银行抽贷", "中", "提前续贷谈判", "财务"),
        ("环保处罚未决事项", "中", "专项法律意见", "法务"),
        ("数据迁移失败", "低", "双跑并行两周", "IT"),
    ], start=1)
] + [
    "二十、例会决议摘录\n",
    "第 6 次例会：确认评估机构工作计划；要求补充目标公司前五大客户集中度分析；",
    "第 7 次例会：通过过渡期预算表 v3；第 8 次例会：同意将反垄断申报材料提交时间提前一周。\n",
])

_pad("case_044", [
    "十七、系统权限整改基线（模拟）\n",
] + [
    "- 项{0}：{1}；目标 {2}；验收方式：{3}。\n".format(idx, item, target, check)
    for idx, (item, target, check) in enumerate([
        ("行级权限覆盖", "100% 客户表", "策略扫描工具季度核对"),
        ("导出二次审批", "全部导出动作", "抽样 20 笔回看"),
        ("审计字段到行级", "过滤条件入日志", "日志样例评审"),
        ("离职账号回收", "当日 18:00 前", "与 HR 名单自动比对"),
        ("敏感字段掩码", "生产全量展示层", "前端与报表双端核对"),
    ], start=1)
] + [
    "十八、教训沉淀（管理层摘录）\n",
    "权限模型缺陷是本次事件放大的根本原因；告警只是最后一道防线，",
    "设计期的最小权限与审批链才是第一道防线。复盘材料纳入新员工必修课。\n",
])

_pad("case_045", [
    "二十一、接口与对账需求（模拟）\n",
] + [
    "- 接口{0}：{1}；频率 {2}；超时 {3} 秒；对账方式：{4}。\n".format(idx, name, freq, timeout, recon)
    for idx, (name, freq, timeout, recon) in enumerate([
        ("客流上传", "每 5 分钟", 10, "按日全量核对"),
        ("票款清分", "每 30 分钟", 15, "按班次批次核对"),
        ("设备状态心跳", "每 1 分钟", 5, "阈值告警"),
        ("黑名单下发", "每 15 分钟", 10, "回执比对"),
        ("对账文件下载", "每日 02:00", 30, "MD5 校验"),
    ], start=1)
] + [
    "二十二、联调场地安排：模拟站点 2 座（车站编码 M6-T01/M6-T02）；",
    "联调期 6 周；第 1-2 周单机，第 3-4 周系统级，第 5-6 周压力与故障注入。\n",
])


LONG_FORM_DOCS = (
    "case_031", "case_033", "case_034", "case_039", "case_040",
    "case_044", "case_045", "case_071", "case_072", "case_076",
    "case_077", "case_078", "case_088", "case_090",
)


def _autopad(doc, target=3050):
    """Deterministically extends a long-form document with a neutral review
    appendix (no entity fragments - appending gold text would multiply the
    entity list geometrically through EXTEND re-scanning)."""
    while len(doc["text"]) < target:
        batch = "R" + str(len(doc["text"]))
        appendix = ("\n附：文档复核记录（批次 {batch}）：全文要素与口径经复核继续适用；"
                    "本批次复核范围覆盖前述全部章节；复核结论存档于原文件末页。").format(batch=batch)
        doc["text"] += appendix


def main():
    for case_id, extras in EXTENSIONS.items():
        EXTEND(case_id, "".join(extras))
    by_id = {d["id"]: d for d in DOCS}
    for case_id in LONG_FORM_DOCS:
        if case_id in by_id:
            _autopad(by_id[case_id])
    # Relabel length classes from actual text length (canonical bands).
    for d in DOCS:
        n = len(d["text"])
        d["length_class"] = ("short" if n < 100 else
                             "medium" if n < 500 else
                             "long" if n < 3000 else "very_long")
    assert len(DOCS) == 100, f"expected 100 docs, got {len(DOCS)}"
    ids = [d["id"] for d in DOCS]
    assert len(set(ids)) == 100, "duplicate ids"
    for d in DOCS:
        text = d["text"]
        for ent in d["entities"] + d["semantic_privacy"]:
            assert 0 <= ent["start"] < ent["end"] <= len(text), (d["id"], ent)
            assert text[ent["start"]:ent["end"]] == ent["text"], (d["id"], ent)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", encoding="utf-8") as f:
        for d in DOCS:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # summary
    from collections import Counter
    lang = Counter(d["language"] for d in DOCS)
    lc = Counter(d["length_class"] for d in DOCS)
    long_docs = sum(1 for d in DOCS if len(d["text"]) > 3000)
    ent_count = sum(len(d["entities"]) for d in DOCS)
    redact_count = sum(1 for d in DOCS for e in d["entities"] if e["should_redact"])
    sem_count = sum(len(d["semantic_privacy"]) for d in DOCS)
    sem_sensitive = sum(1 for d in DOCS for s in d["semantic_privacy"] if s["sensitive"])
    pii_free = sum(1 for d in DOCS if d["pii_free"])
    quasi = sum(1 for d in DOCS if d["risk_group"] == "quasi_identifier")
    print(f"corpus written: {FIXTURE_PATH}")
    print(f"language: {dict(lang)}")
    print(f"length classes: {dict(lc)} | docs >3000 chars: {long_docs}")
    print(f"entity golds: {ent_count} (should_redact=true: {redact_count})")
    print(f"semantic golds: {sem_count} (sensitive: {sem_sensitive})")
    print(f"pii_free docs: {pii_free} | quasi_identifier docs: {quasi}")


if __name__ == "__main__":
    main()
