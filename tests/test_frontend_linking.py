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
          const paramClose = content.indexOf(')', start);
          const brace = content.indexOf('{{', paramClose);
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
        eval(extractFunction('scrollElementIntoContainer'));
        eval(extractFunction('activateLinkedElement'));
        const events = [];
        const classList = {
          remove: (name) => events.push(`remove:${name}`),
          add: (name) => events.push(`add:${name}`),
        };
        const target = {
          classList,
          offsetWidth: 20,
          scrollIntoView: (options) => events.push(`scrollIntoView:called`),
          focus: (opts) => events.push(`focus:preventScroll=${opts && opts.preventScroll}`),
          addEventListener: (name) => events.push(`listen:${name}`),
          getBoundingClientRect: () => ({ top: 120, height: 30 }),
        };
        const container = {
          clientHeight: 200,
          scrollTop: 0,
          getBoundingClientRect: () => ({ top: 0, height: 200 }),
          querySelector: (selector) => {
            events.push(`query:${selector}`);
            return target;
          },
          prepend: (node) => events.push(`prepend:${node === target}`),
          scrollTo: (options) => events.push(`scrollTo:${options.top}`),
        };
        // Case 1: moveFirst = true (prepend and scroll to 0)
        const found = activateLinkedElement(container, '[data-entity-index="2"]', true);
        // Case 2: moveFirst = false (internal scrollElementIntoContainer)
        activateLinkedElement(container, '[data-entity-index="2"]', false);
        console.log(JSON.stringify({ found, events }));
        """)

        self.assertTrue(result["found"])
        self.assertIn("prepend:true", result["events"])
        self.assertIn("scrollTo:0", result["events"])
        self.assertNotIn("scrollIntoView:called", result["events"])
        self.assertIn("focus:preventScroll=true", result["events"])
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
        eval(extractFunction('allocateReplacementToken'));
        eval(extractFunction('resetAnnotationInteractionState'));
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

        const crypto = require('crypto');
        global.crypto = {
          subtle: {
            digest: async (algo, data) => crypto.createHash('sha256').update(data).digest()
          }
        };
        const encoder = new (require('util').TextEncoder)();

        eval(extractFunction('shortFingerprint'));
        eval(extractFunction('allocateReplacementToken'));
        eval(extractFunction('resetAnnotationInteractionState'));
        eval(extractFunction('resolveEntity'));
        eval(extractFunction('generateRedacted'));
        eval(extractFunction('startEntityRetarget'));
        eval(extractFunction('cancelEntityRetarget'));
        eval(extractFunction('applyEntityRetarget'));

        (async () => {
          startEntityRetarget('e1');
          const snapshot = { ...state.retargeting.snapshot };
          cancelEntityRetarget(false);
          const restoredAfterCancel = state.entities[0].start === 3 && state.entities[0].end === 5;

          startEntityRetarget('e1');
          await applyEntityRetarget({ start: 7, end: 15, text: '电话13800' });
          const overlapRejected = state.retargeting !== null;

          await applyEntityRetarget({ start: 0, end: 5, text: '联系人张三' });
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
        })();
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

    def test_full_annotation_state_machine_simulation(self):
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
            scrollTo: (opts) => domEvents.push(`scrollTo:${opts && opts.top}`),
            getBoundingClientRect: () => ({ top: 100, bottom: 120, height: 20, left: 0, right: 100, width: 100 }),
            scrollIntoView: (opts) => domEvents.push(`scroll:${opts.block}`),
            focus: (opts) => domEvents.push(`dom:focus:preventScroll=${opts && opts.preventScroll}`),
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
        global.elements = elements;
        global.window = { matchMedia: () => ({ matches: false }), getSelection: () => null };

        function toast(msg) { domEvents.push(`toast:${msg}`); }
        function setCounter() {}
        function showNotice() {}
        function updateRedactedActionAvailability() {}
        function updateVaultSummary() {}

        eval(extractFunction("shortFingerprint"));
        eval(extractFunction("allocateReplacementToken"));
        eval(extractFunction("resetAnnotationInteractionState"));
        eval(extractFunction("invalidateRedactedState"));
        eval(extractFunction("prepareEntities"));
        eval(extractFunction("maskPreview"));
        eval(extractFunction("buildRedactedSegments"));
        eval(extractFunction("normalizeSourceSelection"));
        eval(extractFunction("scrollElementIntoContainer"));
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
          await applyEntityRetarget(newRange);
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
    def test_unicode_safe_copy_with_emoji_offsets(self):
        result = self.run_node("""
        const crypto = require('crypto');
        global.crypto = {
          subtle: {
            digest: async (algo, data) => crypto.createHash('sha256').update(data).digest()
          }
        };
        const encoder = new (require('util').TextEncoder)();
        let entitySequence = 0;
        const state = { source: '', entities: [], vault: [] };
        const elements = {
          redactedText: { value: '' },
          redactedCounter: { textContent: '' },
          copyRedactedButton: { disabled: true },
          copyPromptButton: { disabled: true },
          exportVaultButton: { disabled: true },
          vaultSummary: { textContent: '' },
          activeVaultBadge: { textContent: '' },
          selectionPopover: { hidden: true, replaceChildren: () => {}, classList: { remove: () => {} } }
        };
        function renderEntities() {}
        function renderRedactedPreview() {}
        function hideSelectionPopover() {}
        function setCounter() {}
        function showNotice() {}
        function updateRedactedActionAvailability() {}

        eval(extractFunction('shortFingerprint'));
        eval(extractFunction('allocateReplacementToken'));
        eval(extractFunction('resetAnnotationInteractionState'));
        eval(extractFunction('invalidateRedactedState'));
        eval(extractFunction('prepareEntities'));
        eval(extractFunction('generateRedacted'));

        (async () => {
          const testCases = [
            // 1. Emoji 😀 (1 codepoint, 2 code units)
            {
              text: '😀张三的电话是13800138000',
              raw: [{ type: 'PHONE', label: '电话号码', text: '13800138000', start: 7, end: 18, start_utf16: 8, end_utf16: 19 }]
            },
            // 2. Regional flag 🇨🇳 (2 codepoints, 4 code units)
            {
              text: '🇨🇳张三的电话是13800138000',
              raw: [{ type: 'PHONE', label: '电话号码', text: '13800138000', start: 8, end: 19, start_utf16: 10, end_utf16: 21 }]
            },
            // 3. ZWJ emoji 👨‍👩‍👧‍👦 (7 codepoints, 11 code units)
            {
              text: '👨\\u200d👩\\u200d👧\\u200d👦姓名：张三电话13800138000',
              raw: [
                { type: 'CN_NAME', label: '姓名', text: '张三', start: 10, end: 12, start_utf16: 14, end_utf16: 16 },
                { type: 'PHONE', label: '电话号码', text: '13800138000', start: 14, end: 25, start_utf16: 18, end_utf16: 29 }
              ]
            },
            // 4. Astral plane character 𠀀 (U+20000, 1 codepoint, 2 code units)
            {
              text: 'A𠀀B王五电话13800138000',
              raw: [
                { type: 'CN_NAME', label: '姓名', text: '王五', start: 3, end: 5, start_utf16: 4, end_utf16: 6 },
                { type: 'PHONE', label: '电话号码', text: '13800138000', start: 7, end: 18, start_utf16: 8, end_utf16: 19 }
              ]
            },
            // 5. Combining mark é (e + \\u0301, 2 codepoints, 2 code units)
            {
              text: 'e\\u0301张三电话13800138000',
              raw: [{ type: 'PHONE', label: '电话号码', text: '13800138000', start: 5, end: 16, start_utf16: 5, end_utf16: 16 }]
            },
            // 6. Arabic / RTL text
            {
              text: 'مرحبا 13800138000 شكرا',
              raw: [{ type: 'PHONE', label: '电话号码', text: '13800138000', start: 6, end: 17, start_utf16: 6, end_utf16: 17 }]
            }
          ];

          const outputs = [];
          for (const tc of testCases) {
            state.source = tc.text;
            state.entities = await prepareEntities(tc.raw);
            generateRedacted();
            outputs.push({
              source: tc.text,
              redacted: elements.redactedText.value,
              entities: state.entities
            });
          }
          console.log(JSON.stringify(outputs));
        })();
        """)
        # Verify Case 1 (😀): starts with "😀张三的电话是" and ends with replacement token, NO trailing "0"!
        c1 = result[0]
        self.assertTrue(c1["redacted"].startswith("😀张三的电话是⟦电话号码_"))
        self.assertTrue(c1["redacted"].endswith("⟧"))
        self.assertFalse(c1["redacted"].endswith("0"))

        # Verify Case 2 (🇨🇳): starts with "🇨🇳张三的电话是" and ends with replacement token, NO trailing "0"!
        c2 = result[1]
        self.assertTrue(c2["redacted"].startswith("🇨🇳张三的电话是⟦电话号码_"))
        self.assertTrue(c2["redacted"].endswith("⟧"))
        self.assertFalse(c2["redacted"].endswith("0"))

        # Verify Case 3 (👨‍👩‍👧‍👦): no broken surrogate, no replacement character, correct tokens
        c3 = result[2]
        self.assertNotIn("\ufffd", c3["redacted"])
        self.assertIn("⟦姓名_", c3["redacted"])
        self.assertIn("⟦电话号码_", c3["redacted"])

        # Verify Case 4 (𠀀): astral plane preserved
        c4 = result[3]
        self.assertTrue(c4["redacted"].startswith("A𠀀B⟦姓名_"))
        self.assertNotIn("\ufffd", c4["redacted"])

        # Verify Case 6 (Arabic):
        c6 = result[5]
        self.assertIn("مرحبا", c6["redacted"])
        self.assertIn("شكرا", c6["redacted"])

    def test_unicode_vault_restore_roundtrip(self):
        result = self.run_node("""
        const crypto = require('crypto');
        global.crypto = {
          subtle: {
            digest: async (algo, data) => crypto.createHash('sha256').update(data).digest()
          }
        };
        const encoder = new (require('util').TextEncoder)();
        let entitySequence = 0;
        const state = { source: '', entities: [], vault: [] };
        const elements = {
          redactedText: { value: '' },
          redactedCounter: { textContent: '' },
          copyRedactedButton: { disabled: true },
          copyPromptButton: { disabled: true },
          exportVaultButton: { disabled: true },
          vaultSummary: { textContent: '' },
          activeVaultBadge: { textContent: '' },
          selectionPopover: { hidden: true, replaceChildren: () => {}, classList: { remove: () => {} } }
        };
        function renderEntities() {}
        function renderRedactedPreview() {}
        function hideSelectionPopover() {}
        function setCounter() {}
        function showNotice() {}
        function updateRedactedActionAvailability() {}

        eval(extractFunction('shortFingerprint'));
        eval(extractFunction('allocateReplacementToken'));
        eval(extractFunction('resetAnnotationInteractionState'));
        eval(extractFunction('invalidateRedactedState'));
        eval(extractFunction('prepareEntities'));
        eval(extractFunction('generateRedacted'));

        (async () => {
          const original = '😀张三联系了A𠀀B王五，电话是13800138000。';
          state.source = original;
          state.entities = await prepareEntities([
            { type: 'CN_NAME', label: '姓名', text: '张三', start: 1, end: 3, start_utf16: 2, end_utf16: 4 },
            { type: 'CN_NAME', label: '姓名', text: '王五', start: 9, end: 11, start_utf16: 11, end_utf16: 13 },
            { type: 'PHONE', label: '电话号码', text: '13800138000', start: 15, end: 26, start_utf16: 17, end_utf16: 28 }
          ]);
          generateRedacted();
          const safeCopy = elements.redactedText.value;

          // Simulate AI reply repeating the placeholders
          const aiReply = `收到，我们将尽快联系 ${state.entities[0].replacement} 和 ${state.entities[1].replacement}，确认号码 ${state.entities[2].replacement}。`;

          // Restore using state.vault
          let restored = aiReply;
          state.vault.forEach((item) => {
            restored = restored.replaceAll(item.token, item.value);
          });

          // Also restore safe copy directly
          let fullRestored = safeCopy;
          state.vault.forEach((item) => {
            fullRestored = fullRestored.replaceAll(item.token, item.value);
          });

          console.log(JSON.stringify({
            original,
            safeCopy,
            aiReply,
            restored,
            fullRestored,
            vaultCount: state.vault.length
          }));
        })();
        """)
        self.assertEqual(result["fullRestored"], result["original"])
        self.assertIn("张三", result["restored"])
        self.assertIn("王五", result["restored"])
        self.assertIn("13800138000", result["restored"])
        self.assertEqual(result["vaultCount"], 3)

    def test_manual_duplicate_reuses_existing_token(self):
        result = self.run_node("""
        const crypto = require('crypto');
        global.crypto = {
          subtle: {
            digest: async (algo, data) => crypto.createHash('sha256').update(data).digest()
          }
        };
        const encoder = new (require('util').TextEncoder)();
        let entitySequence = 0;
        const state = { source: '', entities: [], vault: [], pendingSelection: null };
        const notices = [];
        const elements = {
          redactedText: { value: '' },
          redactedCounter: { textContent: '' },
          copyRedactedButton: { disabled: true },
          copyPromptButton: { disabled: true },
          exportVaultButton: { disabled: true },
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
        function showNotice(msg) { if (msg) notices.push(msg); }
        function updateRedactedActionAvailability() {}

        eval(extractFunction('shortFingerprint'));
        eval(extractFunction('allocateReplacementToken'));
        eval(extractFunction('resetAnnotationInteractionState'));
        eval(extractFunction('invalidateRedactedState'));
        eval(extractFunction('prepareEntities'));
        eval(extractFunction('createManualEntity'));
        eval(extractFunction('generateRedacted'));

        (async () => {
          state.source = '张三说你好，张三说再见。';
          // Auto detector only found the first "张三"
          state.entities = await prepareEntities([
            { type: 'CN_NAME', label: '姓名', text: '张三', start: 0, end: 2, start_utf16: 0, end_utf16: 2 }
          ]);
          generateRedacted();
          const autoToken = state.entities[0].replacement;

          // User manually tags the second "张三" at 6..8
          const manualOption = { type: 'CN_NAME', label: '姓名', privacyLevel: 'PL2' };
          await createManualEntity({ text: '张三', start: 6, end: 8 }, manualOption);

          const manualToken = state.entities.find(e => e.start === 6).replacement;
          const redacted = elements.redactedText.value;
          const copyEnabled = !elements.copyRedactedButton.disabled;

          console.log(JSON.stringify({
            autoToken,
            manualToken,
            redacted,
            copyEnabled,
            notices,
            vault: state.vault
          }));
        })();
        """)
        # Both auto and manual entity share the same token
        self.assertEqual(result["autoToken"], result["manualToken"])
        self.assertTrue(result["copyEnabled"])
        self.assertEqual(len(result["vault"]), 1)
        # Redacted text has both replaced by the same token
        token = result["autoToken"]
        expected = f"{token}说你好，{token}说再见。"
        self.assertEqual(result["redacted"], expected)

    def test_generate_redacted_rejects_token_value_collision(self):
        result = self.run_node("""
        let entitySequence = 0;
        const notices = [];
        const state = {
          source: '张三和李四在此。',
          entities: [
            { id: 'e1', type: 'CN_NAME', label: '姓名', text: '张三', start: 0, end: 2, enabled: true, replacement: '⟦姓名_01_TEST⟧' },
            // Tampered/collision: same token but different value '李四'
            { id: 'e2', type: 'CN_NAME', label: '姓名', text: '李四', start: 3, end: 5, enabled: true, replacement: '⟦姓名_01_TEST⟧' }
          ],
          vault: []
        };
        const elements = {
          redactedText: { value: 'old valid text' },
          redactedCounter: { textContent: '' },
          copyRedactedButton: { disabled: false },
          copyPromptButton: { disabled: false },
          exportVaultButton: { disabled: false },
          vaultSummary: { textContent: '' },
          activeVaultBadge: { textContent: '' }
        };
        function renderRedactedPreview() {}
        function setCounter() {}
        function showNotice(msg, isError) { notices.push({ msg, isError }); }

        eval(extractFunction('invalidateRedactedState'));
        eval(extractFunction('generateRedacted'));

        generateRedacted();

        console.log(JSON.stringify({
          copyDisabled: elements.copyRedactedButton.disabled,
          promptDisabled: elements.copyPromptButton.disabled,
          exportDisabled: elements.exportVaultButton.disabled,
          redactedValue: elements.redactedText.value,
          vaultLen: state.vault.length,
          notices
        }));
        """)
        self.assertTrue(result["copyDisabled"])
        self.assertTrue(result["promptDisabled"])
        self.assertTrue(result["exportDisabled"])
        self.assertEqual(result["redactedValue"], "")
        self.assertEqual(result["vaultLen"], 0)
        self.assertTrue(any("同一占位符对应了不同原文内容" in n["msg"] for n in result["notices"]))

    def test_generate_redacted_failure_disables_copy(self):
        result = self.run_node("""
        let entitySequence = 0;
        const notices = [];
        const state = {
          source: '重叠测试样例。',
          entities: [
            // Overlapping spans: 0..4 and 2..6
            { id: 'e1', type: 'GENERIC_PRIVACY', label: '隐私条目', text: '重叠测试', start: 0, end: 4, enabled: true, replacement: '⟦隐私条目_01_A⟧' },
            { id: 'e2', type: 'GENERIC_PRIVACY', label: '隐私条目', text: '测试样例', start: 2, end: 6, enabled: true, replacement: '⟦隐私条目_02_B⟧' }
          ],
          vault: [{ token: 'old', value: 'val' }]
        };
        const elements = {
          redactedText: { value: 'previously generated safe copy' },
          redactedCounter: { textContent: '' },
          copyRedactedButton: { disabled: false },
          copyPromptButton: { disabled: false },
          exportVaultButton: { disabled: false },
          vaultSummary: { textContent: '3 个加密映射仅保留在当前页面' },
          activeVaultBadge: { textContent: '当前会话：3 个映射' }
        };
        function renderRedactedPreview() {}
        function setCounter() {}
        function showNotice(msg, isError) { notices.push({ msg, isError }); }

        eval(extractFunction('invalidateRedactedState'));
        eval(extractFunction('generateRedacted'));

        generateRedacted();

        console.log(JSON.stringify({
          copyDisabled: elements.copyRedactedButton.disabled,
          promptDisabled: elements.copyPromptButton.disabled,
          exportDisabled: elements.exportVaultButton.disabled,
          redactedValue: elements.redactedText.value,
          vaultLen: state.vault.length,
          summary: elements.vaultSummary.textContent,
          notices
        }));
        """)
        self.assertTrue(result["copyDisabled"])
        self.assertTrue(result["promptDisabled"])
        self.assertTrue(result["exportDisabled"])
        self.assertEqual(result["redactedValue"], "")
        self.assertEqual(result["vaultLen"], 0)
        self.assertIn("当前脱敏结果无效", result["summary"])
        self.assertTrue(any("重叠的隐私条目范围" in n["msg"] for n in result["notices"]))

    def test_detect_resets_retarget_state(self):
        result = self.run_node("""
        const state = {
          source: '旧文本',
          entities: [{ id: 'e1', text: '旧', start: 0, end: 1, enabled: true, replacement: '⟦旧_01_A⟧' }],
          retargeting: { entity: {}, snapshot: {} },
          pendingSelection: { start: 0, end: 1, text: '旧' }
        };
        const elements = {
          sourceText: { value: '新文本内容' },
          detectButton: { disabled: false, innerHTML: '' },
          redactedPreview: { classList: { remove: () => {} } },
          entityList: { classList: { remove: () => {} } },
          selectionPopover: { hidden: false, replaceChildren: () => {}, classList: { remove: () => {} } }
        };
        function clearBrowserSelection() {}
        function hideSelectionPopover() {}
        function showNotice() {}
        function renderEntities() {}
        function generateRedacted() {}
        function updateRedactedActionAvailability() {}
        async function api(path, opts) {
          return { entities: [], policy_level: 'PL2', processing_ms: 5 };
        }
        async function prepareEntities(e) { return e; }

        eval(extractFunction('resetAnnotationInteractionState'));
        eval(extractFunction('detect'));

        (async () => {
          await detect();
          console.log(JSON.stringify({
            retargeting: state.retargeting,
            pendingSelection: state.pendingSelection,
            source: state.source
          }));
        })();
        """)
        self.assertIsNone(result["retargeting"])
        self.assertIsNone(result["pendingSelection"])
        self.assertEqual(result["source"], "新文本内容")

    def test_clear_all_resets_annotation_state(self):
        result = self.run_node("""
        const state = {
          source: '测试文本',
          entities: [{ id: 'e1', text: '测试', start: 0, end: 2 }],
          vault: [{ token: 't', value: 'v' }],
          retargeting: { entity: {} },
          pendingSelection: { text: '选区' }
        };
        const elements = {
          sourceText: { value: '测试文本' },
          redactedText: { value: '脱敏文本' },
          replyText: { value: '回复' },
          restoredText: { value: '恢复' },
          sourceCounter: {},
          redactedCounter: {},
          replyCounter: {},
          restoredCounter: {},
          copyRedactedButton: { disabled: false },
          copyPromptButton: { disabled: false },
          copyRestoredButton: { disabled: false },
          exportVaultButton: { disabled: false },
          vaultSummary: { textContent: '3 映射' },
          activeVaultBadge: { textContent: '3 映射' },
          restoreReport: { hidden: false },
          redactedPreview: { classList: { remove: () => {} } },
          entityList: { classList: { remove: () => {} } },
          selectionPopover: { hidden: false, replaceChildren: () => {}, classList: { remove: () => {} } }
        };
        function clearBrowserSelection() {}
        function hideSelectionPopover() {}
        function renderEntities() {}
        function renderRedactedPreview() {}
        function setCounter() {}
        function showNotice() {}
        function toast() {}
        function updateRedactedActionAvailability() {}

        eval(extractFunction('resetAnnotationInteractionState'));
        eval(extractFunction('clearAll'));

        clearAll();

        console.log(JSON.stringify({
          source: state.source,
          entitiesLen: state.entities.length,
          vaultLen: state.vault.length,
          retargeting: state.retargeting,
          pendingSelection: state.pendingSelection,
          copyRedactedDisabled: elements.copyRedactedButton.disabled
        }));
        """)
        self.assertEqual(result["source"], "")
        self.assertEqual(result["entitiesLen"], 0)
        self.assertEqual(result["vaultLen"], 0)
        self.assertIsNone(result["retargeting"])
        self.assertIsNone(result["pendingSelection"])
        self.assertTrue(result["copyRedactedDisabled"])

    def test_disabled_entity_does_not_block_manual_annotation(self):
        result = self.run_node("""
        eval(extractFunction('normalizeSourceSelection'));

        const source = '联系人张三先生';
        // Case A: entity enabled=false
        const entitiesDisabled = [
          { id: 'e1', label: '姓名', text: '张三', start: 3, end: 5, enabled: false }
        ];
        const resA = normalizeSourceSelection(source, 3, 5, entitiesDisabled);

        // Case B: entity enabled=true
        const entitiesEnabled = [
          { id: 'e1', label: '姓名', text: '张三', start: 3, end: 5, enabled: true }
        ];
        const resB = normalizeSourceSelection(source, 3, 5, entitiesEnabled);

        console.log(JSON.stringify({ resA, resB }));
        """)
        # When disabled: selection over 3..5 is allowed
        self.assertIsNotNone(result["resA"])
        self.assertEqual(result["resA"]["start"], 3)
        self.assertEqual(result["resA"]["end"], 5)
        self.assertEqual(result["resA"]["text"], "张三")

        # When enabled: selection over 3..5 is rejected as overlap
        self.assertIsNone(result["resB"])

    def test_right_click_mouseup_does_not_close_context_menu(self):
        """Right click mouseup (button !== 0) must not trigger selection handling or dismiss context menu."""
        app_code = self.app_js.read_text(encoding="utf-8")
        self.assertIn("if (event.button !== 0) return;", app_code)

        result = self.run_node("""
        let selectionHandled = false;
        function handlePreviewSelection() { selectionHandled = true; }

        let mouseUpHandler = null;
        const mockPreview = {
          addEventListener: (event, handler) => {
            if (event === 'mouseup') mouseUpHandler = handler;
          }
        };

        // Simulate app.js listener registration logic
        mockPreview.addEventListener('mouseup', (event) => {
          if (event.button !== 0) return;
          handlePreviewSelection();
        });

        // Test right-click (button === 2)
        mouseUpHandler({ button: 2 });
        const rightClickIgnored = !selectionHandled;

        // Test left-click (button === 0)
        mouseUpHandler({ button: 0 });
        const leftClickHandled = selectionHandled;

        console.log(JSON.stringify({ rightClickIgnored, leftClickHandled }));
        """)
        self.assertTrue(result["rightClickIgnored"])
        self.assertTrue(result["leftClickHandled"])

    def test_review_highlight_has_no_horizontal_translation(self):
        """Keyframe entity-link-pulse must not use horizontal translateX, preventing card overflow."""
        pulse_start = self.styles.find("@keyframes entity-link-pulse")
        self.assertNotEqual(pulse_start, -1)
        pulse_end = self.styles.find("}", self.styles.find("70%", pulse_start)) + 1
        pulse_css = self.styles[pulse_start:pulse_end]
        self.assertNotIn("transform", pulse_css)
        self.assertNotIn("translateX", pulse_css)
        self.assertIn("border-color", pulse_css)
        self.assertIn("box-shadow", pulse_css)

    def test_retarget_source_has_no_dashed_outline(self):
        """Range retarget source marker must not have misleading dashed outline."""
        retarget_start = self.styles.find(".redacted-retarget-source")
        self.assertNotEqual(retarget_start, -1)
        retarget_end = self.styles.find("}", retarget_start) + 1
        retarget_css = self.styles[retarget_start:retarget_end]
        self.assertNotIn("dashed", retarget_css)
        self.assertNotIn("outline", retarget_css)
        self.assertIn("#ffe999", retarget_css)

    def test_mask_workspace_has_fixed_desktop_height(self):
        """Workspace 3 columns in #maskView must share equal desktop fixed height of 640px and responsive reset."""
        self.assertIn("#maskView .workspace-grid > .panel { height: 640px; min-height: 0; }", self.styles)
        self.assertIn("#maskView #sourceText { min-height: 0; flex: 1 1 0; overflow: auto; resize: none; }", self.styles)
        self.assertIn("#maskView .redacted-preview { min-height: 0; flex: 1 1 0; overflow-y: auto; overflow-x: hidden; }", self.styles)
        # Verify responsive overrides reset height
        self.assertIn("#maskView .workspace-grid > .panel { height: auto; min-height: 480px; }", self.styles)
        self.assertIn("#maskView .workspace-grid > .panel { height: auto; min-height: 440px; }", self.styles)

    def test_restore_workspace_not_forced_to_fixed_height(self):
        """Restore workspace (.restore-grid) must NOT be locked to height: 640px."""
        self.assertNotIn("#restoreView .restore-grid > .panel { height: 640px", self.styles)
        self.assertNotIn(".restore-grid .panel { height: 640px", self.styles)
        self.assertIn(".restore-grid .panel { min-height: 490px;", self.styles)

    def test_entity_list_uses_vertical_only_overflow(self):
        """Review panel entity list must strictly use overflow-y: auto and overflow-x: hidden."""
        self.assertIn(".entity-list { display: flex; flex-direction: column; gap: 9px; overflow-y: auto; overflow-x: hidden; max-height: 435px;", self.styles)
        self.assertIn("#maskView .entity-list { flex: 1 1 0; min-height: 0; max-height: none; }", self.styles)

    def test_link_navigation_does_not_use_scroll_into_view(self):
        """Bi-directional linking and retargeting must use scrollElementIntoContainer without scrollIntoView."""
        app_code = self.app_js.read_text(encoding="utf-8")
        self.assertFalse(any("scrollIntoView" in line for line in app_code.splitlines() if not line.strip().startswith("//")))
        self.assertIn("function scrollElementIntoContainer(container, target, behavior", app_code)

        result = self.run_node("""
        eval(extractFunction('scrollElementIntoContainer'));

        let scrolledTo = null;
        const container = {
          scrollTop: 100,
          clientHeight: 500,
          getBoundingClientRect: () => ({ top: 50, bottom: 550 }),
          scrollTo: (opts) => { scrolledTo = opts; }
        };
        const target = {
          clientHeight: 80,
          getBoundingClientRect: () => ({ top: 250, bottom: 330 })
        };

        scrollElementIntoContainer(container, target, 'smooth');

        console.log(JSON.stringify({
          scrolledTo,
          expectedTop: 100 + (250 - 50) - (500 / 2) + 40
        }));
        """)
        self.assertIsNotNone(result["scrolledTo"])
        self.assertEqual(result["scrolledTo"]["behavior"], "smooth")
        self.assertEqual(result["scrolledTo"]["top"], result["expectedTop"])


if __name__ == "__main__":
    unittest.main()
