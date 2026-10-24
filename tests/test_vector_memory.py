"""向量记忆索引与语义检索的单元测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402
import vector_memory  # noqa: E402
from memory import (  # noqa: E402
    JsonMemory,
    format_memory_context,
    remember_entry,
    search_memory_entry,
)
from sessions import SessionStore  # noqa: E402
from tool_context import ToolContext  # noqa: E402
from vector_memory import VectorMemoryStore, embed_text  # noqa: E402


def _entry(value: str, updated_at: str = "2026-01-01T00:00:00") -> dict:
    return {"value": value, "updated_at": updated_at}


def _clear_geo_caches() -> None:
    tools._reverse_geocode_cached.cache_clear()
    tools._reverse_lookup.cache_clear()
    tools._bigdatacloud_reverse.cache_clear()
    tools._reverse_geocode_detail_cached.cache_clear()
    tools._forward_geocode_cached.cache_clear()


class TestEmbedding(unittest.TestCase):
    def test_embedding_is_deterministic_and_normalized(self):
        first = embed_text("杭州的天气怎么样")
        second = embed_text("杭州的天气怎么样")
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0, places=5)

    def test_similar_text_scores_higher_than_unrelated(self):
        query = embed_text("我喜欢喝什么茶")
        related = vector_memory._dot(
            query, embed_text(vector_memory.entry_text("茶", "用户喜欢龙井"))
        )
        unrelated = vector_memory._dot(
            query, embed_text(vector_memory.entry_text("项目", "手写 ReAct Agent"))
        )
        self.assertGreater(related, unrelated)


class VectorCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.db_path = self.root / "vector_memory.db"
        vector_memory.reset_store_cache()

    def tearDown(self) -> None:
        vector_memory.reset_store_cache()
        self._temp.cleanup()


class TestVectorStore(VectorCase):
    def test_sync_and_search(self):
        store = VectorMemoryStore(self.db_path)
        embedded = store.sync(
            "global",
            "",
            {
                "茶": _entry("用户喜欢龙井", "2026-01-01T00:00:00"),
                "城市": _entry("常住杭州", "2026-01-02T00:00:00"),
                "项目": _entry("手写 ReAct Agent", "2026-01-03T00:00:00"),
            },
        )
        self.assertEqual(embedded, 3)

        hits = store.search("我喜欢喝什么茶", [("global", "")], limit=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["key"], "茶")
        self.assertGreater(hits[0]["score"], 0)

    def test_namespaces_are_isolated(self):
        store = VectorMemoryStore(self.db_path)
        store.sync("global", "", {"城市": _entry("常住杭州")})
        store.sync("session", "a" * 32, {"计划": _entry("周末去杭州")})

        session_hits = store.search("杭州", [("session", "a" * 32)])
        self.assertEqual({hit["key"] for hit in session_hits}, {"计划"})

        global_hits = store.search("杭州", [("global", "")])
        self.assertEqual({hit["key"] for hit in global_hits}, {"城市"})

        both = store.search("杭州", [("session", "a" * 32), ("global", "")])
        self.assertEqual({hit["key"] for hit in both}, {"计划", "城市"})

    def test_sync_prunes_deleted_entries(self):
        store = VectorMemoryStore(self.db_path)
        store.sync("global", "", {"a": _entry("苹果"), "b": _entry("香蕉")})
        self.assertEqual(store.stats()["global"], 2)

        store.sync("global", "", {"a": _entry("苹果")})
        self.assertEqual(store.stats()["global"], 1)

    def test_index_survives_new_store_instance(self):
        VectorMemoryStore(self.db_path).sync("global", "", {"茶": _entry("龙井")})
        hits = VectorMemoryStore(self.db_path).search("龙井", [("global", "")])
        self.assertEqual(hits[0]["key"], "茶")


class TestMemorySearch(VectorCase):
    def setUp(self) -> None:
        super().setUp()
        self.global_file = self.root / "memory.json"
        self.store = SessionStore(self.root / "sessions")
        self.session = self.store.create()
        self.context = ToolContext(
            session_id=self.session["id"],
            session_store=self.store,
            default_scope="session",
        )
        self._global_patch = patch.object(tools, "GLOBAL_MEMORY_FILE", self.global_file)
        self._global_patch.start()

    def tearDown(self) -> None:
        self._global_patch.stop()
        super().tearDown()

    def test_search_memory_tool_uses_session_scope(self):
        tools.run_tool(
            "remember",
            {"key": "饮料偏好", "value": "用户喜欢龙井茶"},
            context=self.context,
        )
        tools.run_tool(
            "remember",
            {"key": "常住城市", "value": "杭州", "scope": "global"},
            context=self.context,
        )

        result = tools.run_tool(
            "search_memory", {"query": "我喜欢喝什么茶"}, context=self.context
        )
        self.assertIn("饮料偏好", result)
        self.assertIn("龙井茶", result)
        self.assertNotIn("常住城市", result)

    def test_search_memory_all_scopes(self):
        tools.run_tool(
            "remember",
            {"key": "城市", "value": "杭州", "scope": "global"},
            context=self.context,
        )
        result = tools.run_tool(
            "search_memory",
            {"query": "杭州", "scope": "all"},
            context=self.context,
        )
        self.assertIn("[全局] 城市", result)

    def test_search_ignores_deleted_memory(self):
        remember_entry(
            JsonMemory(self.global_file), "临时", "这条稍后删除"
        )
        backend = JsonMemory(self.global_file)
        self.assertTrue(search_memory_entry(backend, "删除"))
        backend.remove("临时")
        self.assertEqual(search_memory_entry(backend, "删除"), [])

    def test_keyword_fallback_when_vectors_disabled(self):
        remember_entry(JsonMemory(self.global_file), "茶", "龙井")
        with patch("memory.VECTOR_MEMORY_ENABLED", False):
            hits = search_memory_entry(JsonMemory(self.global_file), "茶")
        self.assertEqual(hits[0]["key"], "茶")

    def test_keyword_fallback_when_index_fails(self):
        remember_entry(JsonMemory(self.global_file), "茶", "龙井")
        backend = JsonMemory(self.global_file)
        with patch.object(JsonMemory, "vector_store", side_effect=OSError("磁盘只读")):
            hits = search_memory_entry(backend, "茶")
        self.assertEqual(hits[0]["key"], "茶")

    def test_memory_context_reranks_by_query(self):
        memories = {
            f"记录{i}": _entry(
                "这是一条与问题无关的普通记录。" * 6,
                f"2026-01-{i:02d}T00:00:00",
            )
            for i in range(1, 20)
        }
        memories["研究主题"] = _entry(
            "用户在研究量子计算与相干时间。", "2025-01-01T00:00:00"
        )

        without_query = format_memory_context(memories, {}, max_chars=320)
        self.assertNotIn("研究主题", without_query)

        with_query = format_memory_context(
            memories, {}, max_chars=320, query="量子计算的相干时间是什么"
        )
        self.assertIn("研究主题", with_query)


class TestLocationTool(VectorCase):
    class FakeResponse:
        def __init__(self, data, status_code: int = 200) -> None:
            self._data = data
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise tools.requests.HTTPError(f"HTTP {self.status_code}")

        def json(self) -> dict:
            return self._data

    def setUp(self) -> None:
        super().setUp()
        _clear_geo_caches()
        self.ipwho = self.FakeResponse(
            {
                "success": True,
                "ip": "8.8.8.8",
                "city": "杭州",
                "region": "浙江",
                "country": "中国",
                "country_code": "CN",
                "latitude": 30.2741,
                "longitude": 120.1551,
                "timezone": {"id": "Asia/Shanghai"},
                "connection": {"isp": "Example Telecom", "org": "Example"},
            }
        )

    def test_get_location(self):
        with patch.object(tools.requests, "get", return_value=self.ipwho):
            text = tools.get_location()
        self.assertIn("中国（CN），浙江，杭州", text)
        self.assertIn("30.2741, 120.1551", text)
        self.assertIn("Asia/Shanghai", text)
        self.assertIn("ipwho.is", text)
        self.assertIn("不是 GPS", text)

    def test_gps_location_uses_browser_coordinates(self):
        nominatim = self.FakeResponse(
            {
                "address": {
                    "country": "中国",
                    "state": "浙江省",
                    "city": "杭州市",
                }
            }
        )
        context = ToolContext(
            user_location={
                "latitude": 30.2741,
                "longitude": 120.1551,
                "accuracy": 12,
            }
        )
        with patch.object(tools.requests, "get", return_value=nominatim):
            text = tools.get_location(context=context)

        self.assertIn("设备 GPS/浏览器定位", text)
        self.assertIn("30.274100, 120.155100", text)
        self.assertIn("中国，浙江省，杭州市", text)
        self.assertIn("±12 米", text)
        self.assertIn("不会自动写入长期记忆", text)

    def test_gps_keeps_coordinates_when_reverse_geocode_fails(self):
        context = ToolContext(
            user_location={
                "latitude": 31.2304,
                "longitude": 121.4737,
                "accuracy": 1200,
            }
        )
        with patch.object(
            tools.requests, "get", side_effect=tools.requests.ConnectionError("断网")
        ):
            text = tools.get_location(context=context)

        self.assertIn("31.230400, 121.473700", text)
        self.assertIn("反向地理编码失败", text)
        self.assertIn("精度较低", text)

    def test_location_requests_user_permission(self):
        calls = []

        def requester():
            calls.append(True)
            return {"latitude": 30.2741, "longitude": 120.1551, "accuracy": 20}

        nominatim = self.FakeResponse(
            {"address": {"country": "中国", "state": "浙江省", "city": "杭州市"}}
        )
        context = ToolContext(request_user_location=requester)
        with patch.object(tools.requests, "get", return_value=nominatim):
            text = tools.get_location(context=context)

        self.assertEqual(calls, [True])
        self.assertIn("设备 GPS/浏览器定位", text)
        self.assertIn("30.274100, 120.155100", text)

    def test_location_permission_denied_falls_back_to_ip(self):
        context = ToolContext(request_user_location=lambda: None)
        with patch.object(tools.requests, "get", return_value=self.ipwho):
            text = tools.get_location(context=context)

        self.assertIn("公网 IP", text)
        self.assertIn("用户未授权浏览器定位", text)
        self.assertNotIn("设备 GPS/浏览器定位", text)

    def test_explicit_ip_overrides_gps(self):
        context = ToolContext(
            user_location={"latitude": 30.2741, "longitude": 120.1551}
        )
        with patch.object(tools.requests, "get", return_value=self.ipwho):
            text = tools.get_location("8.8.8.8", context=context)

        self.assertIn("公网 IP", text)
        self.assertNotIn("设备 GPS/浏览器定位", text)

    def test_private_ip_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            tools.get_location("192.168.1.10")
        self.assertIn("内网", str(context.exception))

    def test_fallback_provider(self):
        ipapi = self.FakeResponse(
            {
                "ip": "8.8.8.8",
                "city": "杭州",
                "region": "浙江",
                "country_name": "中国",
                "country_code": "CN",
                "latitude": 30.27,
                "longitude": 120.15,
                "timezone": "Asia/Shanghai",
                "org": "Example Telecom",
            }
        )
        with patch.object(
            tools.requests,
            "get",
            side_effect=[tools.requests.ConnectionError("超时"), ipapi],
        ):
            text = tools.get_location("8.8.8.8")
        self.assertIn("ipapi.co", text)

    def test_network_error_becomes_observation(self):
        with patch.object(
            tools.requests,
            "get",
            side_effect=tools.requests.ConnectionError("断网"),
        ):
            result = tools.run_tool("get_location", {})
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("IP 定位服务暂时不可用", result)


class TestGeoLookup(VectorCase):
    class FakeResponse:
        def __init__(self, data, status_code: int = 200) -> None:
            self._data = data
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise tools.requests.HTTPError(f"HTTP {self.status_code}")

        def json(self):
            return self._data

    def setUp(self) -> None:
        super().setUp()
        _clear_geo_caches()

    def test_distance_uses_local_haversine(self):
        result = tools.run_tool(
            "geo_lookup",
            {
                "operation": "distance",
                "origin": "30.2741,120.1551",
                "destination": "39.9042,116.4074",
            },
        )
        self.assertIn("直线距离", result)
        self.assertIn("公里", result)
        self.assertIn("方位角", result)
        self.assertIn("中点", result)
        # 杭州到北京的球面距离约 1100-1150 km
        distance = float(result.split("直线距离：")[1].split(" 公里")[0])
        self.assertGreater(distance, 1000)
        self.assertLess(distance, 1200)

    def test_distance_rejects_bad_coordinates(self):
        result = tools.run_tool(
            "geo_lookup",
            {"operation": "distance", "origin": "杭州", "destination": "北京"},
        )
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("纬度,经度", result)

    def test_reverse_geocode_prefers_bigdatacloud(self):
        response = self.FakeResponse(
            {
                "countryName": "中国",
                "principalSubdivision": "浙江省",
                "city": "杭州市",
                "locality": "西湖区",
                "localityInfo": {
                    "administrative": [
                        {"name": "中国", "order": 2},
                        {"name": "浙江省", "order": 5},
                        {"name": "杭州市", "order": 6},
                        {"name": "西湖区", "order": 7},
                    ],
                    "informative": [{"name": "Asia/Shanghai", "order": 3}],
                },
            }
        )
        with patch.object(tools.requests, "get", return_value=response):
            result = tools.run_tool(
                "geo_lookup",
                {"operation": "reverse", "latitude": 30.2741, "longitude": 120.1551},
            )
        self.assertIn("地址：中国，浙江省，杭州市，西湖区", result)
        self.assertIn("30.274100, 120.155100", result)
        self.assertIn("时区：Asia/Shanghai", result)
        self.assertIn("来源：BigDataCloud", result)

    def test_reverse_falls_back_to_nominatim(self):
        nominatim = self.FakeResponse(
            {
                "display_name": "北山街道, 西湖区, 杭州市, 浙江省, 中国",
                "address": {
                    "country": "中国",
                    "state": "浙江省",
                    "city": "杭州市",
                    "suburb": "西湖区",
                    "road": "北山街道",
                },
            }
        )

        def fake_get(url, **kwargs):
            if "bigdatacloud" in url:
                raise tools.requests.ConnectionError("断网")
            return nominatim

        with patch.object(tools.requests, "get", side_effect=fake_get):
            result = tools.run_tool(
                "geo_lookup",
                {"operation": "reverse", "latitude": 31.2304, "longitude": 121.4737},
            )
        self.assertIn("杭州市", result)
        self.assertIn("OpenStreetMap Nominatim", result)

    def test_forward_geocode_returns_candidates(self):
        response = self.FakeResponse(
            {
                "results": [
                    {
                        "name": "西湖",
                        "latitude": 30.2448,
                        "longitude": 120.1501,
                        "country": "中国",
                        "admin1": "浙江省",
                        "feature_code": "LK",
                    },
                    {
                        "name": "西湖区",
                        "latitude": 30.2595,
                        "longitude": 120.1300,
                        "country": "中国",
                        "admin1": "浙江省",
                        "feature_code": "ADM2",
                    },
                ]
            }
        )
        with patch.object(tools.requests, "get", return_value=response):
            result = tools.run_tool(
                "geo_lookup", {"operation": "forward", "query": "西湖"}
            )
        self.assertIn("找到 2 个候选地点", result)
        self.assertIn("30.244800, 120.150100", result)
        self.assertIn("Open-Meteo Geocoding", result)

    def test_forward_falls_back_to_nominatim(self):
        nominatim = self.FakeResponse(
            [
                {
                    "display_name": "西湖, 西湖区, 杭州市, 浙江省, 中国",
                    "lat": "30.2448",
                    "lon": "120.1501",
                    "type": "attraction",
                }
            ]
        )

        def fake_get(url, **kwargs):
            if "geocoding-api.open-meteo.com" in url:
                return self.FakeResponse({"results": []})
            return nominatim

        with patch.object(tools.requests, "get", side_effect=fake_get):
            result = tools.run_tool(
                "geo_lookup", {"operation": "forward", "query": "杭州西湖"}
            )
        self.assertIn("OpenStreetMap Nominatim", result)
        self.assertIn("西湖", result)

    def test_unknown_operation_is_reported(self):
        result = tools.run_tool("geo_lookup", {"operation": "walk"})
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("operation 必须是", result)


if __name__ == "__main__":
    unittest.main()
