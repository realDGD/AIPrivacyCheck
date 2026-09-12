import json
from pathlib import Path
import subprocess
import unittest


PROJECT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_DIR / "packaging" / "ai-privacy-check" / "app" / "server" / "web"


class FrontendEntityLinkingTests(unittest.TestCase):
    def setUp(self):
        self.app_js = WEB_DIR / "app.js"
        self.index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        self.styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    def run_node(self, body: str):
        script = f"""
        const fs = require('fs');
        const content = fs.readFileSync({json.dumps(str(self.app_js))}, 'utf8');

        function extractFunction(name) {{
          let start = content.indexOf(`async function ${{name}}`);
          if (start < 0) start = content.indexOf(`function ${{name}}`);
          if (start < 0) throw new Error(`Function not found: ${{name}}`);
          const brace = content.indexOf('{{', start);
          let depth = 0;
          for (let index = brace; index < content.length; index += 1) {{
            if (content[index] === '{{') depth += 1;
            if (content[index] === '}}') depth -= 1;
            if (depth === 0) return content.slice(start, index + 1);
          }}
          throw new Error(`Unclosed function: ${{name}}`);
        }}

        {body}
        """
        proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        return json.loads(proc.stdout)

    def test_segment_builder_preserves_text_and_entity_links(self):
        result = self.run_node("""
        eval(extractFunction('buildRedactedSegments'));
        const segments = buildRedactedSegments('电话13800138000，邮箱a@example.com', [
          { start: 2, end: 13, enabled: true, replacement: '⟦电话号码_01_5BFE⟧', label: '电话号码' },
          { start: 16, end: 29, enabled: true, replacement: '⟦邮箱_01_9A2C⟧', label: '邮箱' },
        ]);
        console.log(JSON.stringify(segments));
        """)

        self.assertEqual(result, [
            {"kind": "text", "text": "电话", "sourceStart": 0, "sourceEnd": 2},
            {
                "kind": "token",
                "text": "⟦电话号码_01_5BFE⟧",
                "entityIndex": 0,
                "label": "电话号码",
                "sourceStart": 2,
                "sourceEnd": 13,
            },
            {"kind": "text", "text": "，邮箱", "sourceStart": 13, "sourceEnd": 16},
            {
                "kind": "token",
                "text": "⟦邮箱_01_9A2C⟧",
                "entityIndex": 1,
                "label": "邮箱",
                "sourceStart": 16,
                "sourceEnd": 29,
            },
        ])

    def test_retarget_builder_restores_only_target_original_text(self):
        result = self.run_node("""
        eval(extractFunction('buildRedactedSegments'));
        const phone = { start: 2, end: 13, enabled: true, replacement: '⟦电话号码_01_5BFE⟧', label: '电话号码' };
        const email = { start: 16, end: 29, enabled: true, replacement: '⟦邮箱_01_9A2C⟧', label: '邮箱' };
        const segments = buildRedactedSegments('电话13800138000，邮箱a@example.com', [phone, email], phone);
        console.log(JSON.stringify(segments));
        """)

        self.assertEqual(result, [
            {"kind": "text", "text": "电话13800138000，邮箱", "sourceStart": 0, "sourceEnd": 16},
            {
                "kind": "token",
                "text": "⟦邮箱_01_9A2C⟧",
                "entityIndex": 1,
                "label": "邮箱",
                "sourceStart": 16,
                "sourceEnd": 29,
            },
        ])

    def test_source_selection_trims_and_rejects_entity_overlap(self):
        result = self.run_node("""
        eval(extractFunction('normalizeSourceSelection'));
        const source = ' 联系人张三，电话13800138000。 ';
        const phone = { start: 7, end: 20, enabled: true };
        console.log(JSON.stringify({
          plain: normalizeSourceSelection(source, 0, 7, [phone]),
          overlap: normalizeSourceSelection(source, 6, 13, [phone]),
          retarget: normalizeSourceSelection(source, 7, 20, [phone], phone),
        }));
        """)

        self.assertEqual(result["plain"], {"start": 1, "end": 7, "text": "联系人张三，"})
        self.assertIsNone(result["overlap"])
        self.assertEqual(result["retarget"], {"start": 7, "end": 20, "text": "电话13800138000"})

    def test_link_activation_moves_focuses_scrolls_and_pulses(self):
        result = self.run_node("""
        eval(extractFunction('activateLinkedElement'));
        const events = [];
        const classList = {
          remove: (name) => events.push(`remove:${name}`),
          add: (name) => events.push(`add:${name}`),
        };
        const target = {
          classList,
          offsetWidth: 20,
          scrollIntoView: (options) => events.push(`scroll:${options.block}`),
          focus: () => events.push('focus'),
          addEventListener: (name) => events.push(`listen:${name}`),
        };
        const container = {
          querySelector: (selector) => {
            events.push(`query:${selector}`);
            return target;
          },
          prepend: (node) => events.push(`prepend:${node === target}`),
        };
        const found = activateLinkedElement(container, '[data-entity-index="2"]', true);
        console.log(JSON.stringify({ found, events }));
        """)

        self.assertTrue(result["found"])
        self.assertIn("prepend:true", result["events"])
        self.assertIn("scroll:center", result["events"])
        self.assertIn("focus", result["events"])
        self.assertIn("add:is-link-target", result["events"])

    def test_source_selection_rejects_empty_whitespace_and_crossing_entities(self):
        result = self.run_node("""
        eval(extractFunction('normalizeSourceSelection'));
        const source = '联系人张三   电话13800138000，邮箱test@example.com。';
        const phone = { start: 10, end: 23, enabled: true };
        const email = { start: 26, end: 42, enabled: true };
        const entities = [phone, email];
        console.log(JSON.stringify({
          empty: normalizeSourceSelection(source, 3, 3, entities),
          whitespace: normalizeSourceSelection(source, 5, 8, entities),
          cross: normalizeSourceSelection(source, 5, 25, entities),
          validGap: normalizeSourceSelection(source, 0, 5, entities),
        }));
        """)
        self.assertIsNone(result["empty"])
        self.assertIsNone(result["whitespace"])
        self.assertIsNone(result["cross"])
        self.assertEqual(result["validGap"]["text"], "联系人张三")

    def test_manual_entity_creation_generic_and_typed(self):
        result = self.run_node("""
        const crypto = require('crypto');
        global.crypto = {
          subtle: {
            digest: async (algo, data) => crypto.createHash('sha256').update(data).digest()
          }
        };
        const encoder = new (require('util').TextEncoder)();
        let entitySequence = 0;
        const state = { entities: [], vault: [], pendingSelection: null };
        const elements = {
          redactedText: { value: '' },
          redactedCounter: { textContent: '' },
          copyRedactedButton: {},
          copyPromptButton: {},
          exportVaultButton: {},
          vaultSummary: { textContent: '' },
          activeVaultBadge: { textContent: '' },
          selectionPopover: { hidden: true, replaceChildren: () => {}, classList: { remove: () => {} } }
        };
        function renderEntities() {}
        function renderRedactedPreview() {}
        function generateRedacted() {}
        function hideSelectionPopover() {}
        function focusReviewEntity() {}
        function toast() {}
        function setCounter() {}
        function updateRedactedActionAvailability() {}
        function updateVaultSummary() {}

        eval(extractFunction('shortFingerprint'));
        eval(extractFunction('createManualEntity'));

        const MANUAL_ENTITY_OPTIONS = [
          { type: 'GENERIC_PRIVACY', label: '隐私条目', privacyLevel: 'PL2' },
          { type: 'CN_NAME', label: '姓名', privacyLevel: 'PL2' },
        ];

        (async () => {
          await createManualEntity({ text: '秘密文字', start: 0, end: 4 }, null);
          await createManualEntity({ text: '李四', start: 10, end: 12 }, MANUAL_ENTITY_OPTIONS[1]);
          console.log(JSON.stringify(state.entities));
        })();
        """)
        self.assertEqual(len(result), 2)
        # First entity is generic "隐私条目"
        self.assertEqual(result[0]["label"], "隐私条目")
        self.assertTrue(result[0]["replacement"].startswith("⟦隐私条目_01_"))
        self.assertEqual(result[0]["source"], "manual")
        self.assertTrue(result[0]["id"].startswith("manual_"))
        # Second entity is typed "姓名"
        self.assertEqual(result[1]["label"], "姓名")
        self.assertTrue(result[1]["replacement"].startswith("⟦姓名_01_"))
        self.assertEqual(result[1]["source"], "manual")

    def test_delete_entity_restores_source_text(self):
        result = self.run_node("""
        let entitySequence = 0;
        const state = {
          source: '张三的电话是13800138000',
          entities: [
            { id: 'e1', label: '姓名', text: '张三', start: 0, end: 2, enabled: true, replacement: '⟦姓名_01_A1B2⟧' },
            { id: 'e2', label: '电话号码', text: '13800138000', start: 6, end: 17, enabled: true, replacement: '⟦电话号码_01_C3D4⟧' }
          ],
          vault: [],
          pendingSelection: null
        };
        const elements = {
          redactedText: { value: '' },
          redactedCounter: { textContent: '' },
          copyRedactedButton: {},
          copyPromptButton: {},
          exportVaultButton: {},
          vaultSummary: { textContent: '' },
          activeVaultBadge: { textContent: '' },
          selectionPopover: { hidden: true, replaceChildren: () => {}, classList: { remove: () => {} } }
        };
        function renderEntities() {}
        function renderRedactedPreview() {}
        function hideSelectionPopover() {}
        function toast() {}
        function setCounter() {}
        function updateRedactedActionAvailability() {}
        function updateVaultSummary() {}

        eval(extractFunction('resolveEntity'));
        eval(extractFunction('generateRedacted'));
        eval(extractFunction('deleteEntity'));

        generateRedacted();
        const before = elements.redactedText.value;
        deleteEntity('e1');
        const after = elements.redactedText.value;
        console.log(JSON.stringify({ before, after, remaining: state.entities.map(e => e.id), vault: state.vault }));
        """)
        self.assertIn("⟦姓名_01_A1B2⟧", result["before"])
        self.assertIn("张三的电话是⟦电话号码_01_C3D4⟧", result["after"])
        self.assertEqual(result["remaining"], ["e2"])
        self.assertEqual(len(result["vault"]), 1)
        self.assertEqual(result["vault"][0]["token"], "⟦电话号码_01_C3D4⟧")

    def test_reselection_mode_and_cancel_snapshot(self):
        result = self.run_node("""
        let entitySequence = 0;
        const state = {
          source: '联系人张三，电话13800138000。',
          entities: [
            { id: 'e1', label: '姓名', text: '张三', start: 3, end: 5, enabled: true, replacement: '⟦姓名_01_A1B2⟧' },
            { id: 'e2', label: '电话号码', text: '13800138000', start: 8, end: 19, enabled: true, replacement: '⟦电话号码_01_C3D4⟧' }
          ],
          vault: [],
          pendingSelection: null,
          retargeting: null
        };
        const elements = {
          redactedText: { value: '' },
          redactedCounter: { textContent: '' },
          redactedPreview: { querySelector: () => null },
          copyRedactedButton: {},
          copyPromptButton: {},
          exportVaultButton: {},
          vaultSummary: { textContent: '' },
          activeVaultBadge: { textContent: '' },
          selectionPopover: { hidden: true, replaceChildren: () => {}, classList: { remove: () => {} } }
        };
        function renderEntities() {}
        function renderRedactedPreview() {}
        function hideSelectionPopover() {}
        function focusReviewEntity() {}
        function toast() {}
        function setCounter() {}
        function updateRedactedActionAvailability() {}
        function updateVaultSummary() {}

        eval(extractFunction('resolveEntity'));
        eval(extractFunction('generateRedacted'));
        eval(extractFunction('startEntityRetarget'));
        eval(extractFunction('cancelEntityRetarget'));
        eval(extractFunction('applyEntityRetarget'));

        startEntityRetarget('e1');
        const snapshot = { ...state.retargeting.snapshot };
        cancelEntityRetarget(false);
        const restoredAfterCancel = state.entities[0].start === 3 && state.entities[0].end === 5;

        startEntityRetarget('e1');
        applyEntityRetarget({ start: 7, end: 15, text: '电话13800' });
        const overlapRejected = state.retargeting !== null;

        applyEntityRetarget({ start: 0, end: 5, text: '联系人张三' });
        const finalEntity = state.entities.find(e => e.id === 'e1');

        console.log(JSON.stringify({
          snapshot,
          restoredAfterCancel,
          overlapRejected,
          finalStart: finalEntity.start,
          finalEnd: finalEntity.end,
          finalText: finalEntity.text,
          finalId: finalEntity.id
        }));
        """)
        self.assertEqual(result["snapshot"], {"start": 3, "end": 5, "text": "张三", "enabled": True, "replacement": "⟦姓名_01_A1B2⟧"})
        self.assertTrue(result["restoredAfterCancel"])
        self.assertTrue(result["overlapRejected"])
        self.assertEqual(result["finalStart"], 0)
        self.assertEqual(result["finalEnd"], 5)
        self.assertEqual(result["finalText"], "联系人张三")
        self.assertEqual(result["finalId"], "e1")

    def test_vault_restore_replaces_manual_tokens(self):
        result = self.run_node("""
        const state = {
          vault: [
            { token: '⟦隐私条目_01_5BFE⟧', value: '绝密数据' },
            { token: '⟦电话号码_01_9A2C⟧', value: '13800138000' }
          ]
        };
        const reply = '分析结果显示 ⟦隐私条目_01_5BFE⟧ 和联系方式 ⟦电话号码_01_9A2C⟧ 已归档。';
        let restored = reply;
        state.vault.forEach((item) => {
          restored = restored.replaceAll(item.token, item.value);
        });
        console.log(JSON.stringify({ restored }));
        """)
        self.assertEqual(result["restored"], "分析结果显示 绝密数据 和联系方式 13800138000 已归档。")

    def test_markup_and_styles_expose_bidirectional_linking(self):
        app_js = self.app_js.read_text(encoding="utf-8")
        self.assertIn('id="redactedPreview"', self.index_html)
        self.assertIn('id="redactedText" hidden', self.index_html)
        self.assertIn("card.dataset.entityIndex", app_js)
        self.assertIn("focusReviewEntity", app_js)
        self.assertIn("focusRedactedEntity", app_js)
        self.assertIn("showEntityActions", app_js)
        self.assertIn("startEntityRetarget", app_js)
        self.assertIn("createManualEntity", app_js)
        self.assertIn("contextmenu", app_js)
        self.assertIn(".redacted-token", self.styles)
        self.assertIn(".selection-popover", self.styles)
        self.assertIn(".redacted-preview.is-retargeting", self.styles)
        self.assertIn(".entity-card.is-link-target", self.styles)

    def test_full_12_step_browser_workflow_simulation(self):
        result = self.run_node("""
        const crypto = require("crypto");
        global.crypto = {
          subtle: {
            digest: async (algo, data) => crypto.createHash("sha256").update(data).digest()
          }
        };
        const encoder = new (require("util").TextEncoder)();

        let entitySequence = 0;
        const state = {
          source: "",
          entities: [],
          vault: [],
          pendingSelection: null,
          retargeting: null
        };

        const domEvents = [];
        function makeElement(tag = "div") {
          const children = [];
          const classes = new Set();
          const dataset = {};
          return {
            tag,
            dataset,
            classList: {
              add: (c) => { classes.add(c); domEvents.push(`add:${c}`); },
              remove: (c) => { classes.delete(c); domEvents.push(`remove:${c}`); },
              contains: (c) => classes.has(c),
              toggle: (c, v) => v ? classes.add(c) : classes.delete(c)
            },
            value: "",
            textContent: "",
            style: {},
            hidden: false,
            disabled: false,
            replaceChildren: (...nodes) => { children.length = 0; children.push(...nodes); },
            append: (...nodes) => children.push(...nodes),
            prepend: (node) => { children.unshift(node); domEvents.push("dom:prepend"); },
            querySelector: (sel) => makeElement("div"),
            scrollIntoView: (opts) => domEvents.push(`scroll:${opts.block}`),
            focus: () => domEvents.push("dom:focus"),
            addEventListener: (ev, fn) => {},
            setAttribute: () => {}
          };
        }

        global.document = {
          createDocumentFragment: () => {
            const list = [];
            return {
              append: (...n) => list.push(...n),
              forEach: (fn) => list.forEach(fn)
            };
          },
          createElement: (tag) => makeElement(tag)
        };

        const elements = {
          sourceText: makeElement("textarea"),
          sourceCounter: makeElement(),
          redactedPreview: makeElement(),
          redactedText: makeElement("textarea"),
          redactedCounter: makeElement(),
          entityList: makeElement(),
          entityEmpty: makeElement(),
          entityCount: makeElement(),
          reviewFooter: makeElement(),
          copyRedactedButton: makeElement("button"),
          copyPromptButton: makeElement("button"),
          copyRestoredButton: makeElement("button"),
          exportVaultButton: makeElement("button"),
          vaultSummary: makeElement(),
          activeVaultBadge: makeElement(),
          detectionNotice: makeElement(),
          selectionPopover: makeElement()
        };

        function toast(msg) { domEvents.push(`toast:${msg}`); }
        function setCounter() {}
        function showNotice() {}
        function updateRedactedActionAvailability() {}
        function updateVaultSummary() {}

        eval(extractFunction("shortFingerprint"));
        eval(extractFunction("prepareEntities"));
        eval(extractFunction("maskPreview"));
        eval(extractFunction("buildRedactedSegments"));
        eval(extractFunction("normalizeSourceSelection"));
        eval(extractFunction("activateLinkedElement"));
        eval(extractFunction("focusReviewEntity"));
        eval(extractFunction("focusRedactedEntity"));
        eval(extractFunction("resolveEntity"));
        eval(extractFunction("clearBrowserSelection"));
        eval(extractFunction("hideSelectionPopover"));
        eval(extractFunction("createManualEntity"));
        eval(extractFunction("showEntityActions"));
        eval(extractFunction("deleteEntity"));
        eval(extractFunction("startEntityRetarget"));
        eval(extractFunction("cancelEntityRetarget"));
        eval(extractFunction("applyEntityRetarget"));
        eval(extractFunction("createSourceSpan"));
        eval(extractFunction("appendTextSegment"));
        eval(extractFunction("generateRedacted"));
        eval(extractFunction("renderRedactedPreview"));
        eval(extractFunction("renderEntities"));

        const MANUAL_ENTITY_OPTIONS = [
          { type: "GENERIC_PRIVACY", label: "隐私条目", privacyLevel: "PL2" },
          { type: "CN_NAME", label: "姓名", privacyLevel: "PL2" },
          { type: "PHONE", label: "电话号码", privacyLevel: "PL2" },
          { type: "CN_ADDRESS", label: "地址", privacyLevel: "PL2" }
        ];

        (async () => {
          // Step 1: Input text
          state.source = "张三的电话号码是 13800001234，住在北京市测试路88号。";
          elements.sourceText.value = state.source;

          // Step 2: Detect
          const rawEntities = [
            { type: "CN_NAME", label: "姓名", text: "张三", start: 0, end: 2 },
            { type: "PHONE", label: "电话号码", text: "13800001234", start: 9, end: 20 },
            { type: "CN_ADDRESS", label: "地址", text: "北京市测试路88号", start: 24, end: 32 }
          ];
          state.entities = await prepareEntities(rawEntities);
          generateRedacted();
          renderEntities();
          renderRedactedPreview();

          // Step 3: Link safe-copy marker -> review card
          const phoneEntity = state.entities[1];
          const step3Ok = focusReviewEntity(phoneEntity.id);

          // Step 4: Link review card -> safe-copy marker
          const step4Ok = focusRedactedEntity(phoneEntity.id);

          // Step 5, 6, 7: Plain text selection & generic manual entity
          const norm = normalizeSourceSelection(state.source, 4, 8, state.entities);
          await createManualEntity(norm, null);
          const manualEnt = state.entities.find(e => e.source === "manual");

          // Step 8, 9: Start retarget
          startEntityRetarget(manualEnt.id);
          const step9Retargeting = state.retargeting !== null;

          // Step 10: Cancel retarget
          cancelEntityRetarget(false);
          const step10Cancelled = state.retargeting === null;

          // Step 11: Re-enter retarget and apply valid new range
          startEntityRetarget(manualEnt.id);
          const newRange = { start: 2, end: 8, text: "的电话号码" };
          applyEntityRetarget(newRange);
          const step11Text = state.entities.find(e => e.id === manualEnt.id).text;

          // Step 12: Delete entity
          deleteEntity(manualEnt.id);
          const step12Count = state.entities.length;
          const step12Restored = elements.redactedText.value;

          console.log(JSON.stringify({
            step3Ok,
            step4Ok,
            manualLabel: manualEnt.label,
            manualToken: manualEnt.replacement,
            step9Retargeting,
            step10Cancelled,
            step11Text,
            step12Count,
            step12Restored
          }));
        })();
        """)
        self.assertTrue(result["step3Ok"])
        self.assertTrue(result["step4Ok"])
        self.assertEqual(result["manualLabel"], "隐私条目")
        self.assertTrue(result["manualToken"].startswith("⟦隐私条目_01_"))
        self.assertTrue(result["step9Retargeting"])
        self.assertTrue(result["step10Cancelled"])
        self.assertEqual(result["step11Text"], "的电话号码")
        self.assertEqual(result["step12Count"], 3)
        self.assertIn("的电话号码", result["step12Restored"])
        self.assertIn("⟦姓名_01_", result["step12Restored"])
        self.assertIn("⟦电话号码_01_", result["step12Restored"])


if __name__ == "__main__":
    unittest.main()
