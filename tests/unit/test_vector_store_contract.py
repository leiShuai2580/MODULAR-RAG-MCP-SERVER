"""Unit tests for VectorStore Factory and Base VectorStore. / VectorStore Factory 和 Base VectorStore 的单元测试。

Test Coverage: / 测试覆盖范围：
- Factory pattern: provider registration, creation, and routing / 工厂模式：provider 注册、创建与路由
- Configuration-driven instantiation / 配置驱动的实例化
- Error handling for unknown/missing providers / 未知或缺失 provider 的错误处理
- Validation logic in BaseVectorStore / BaseVectorStore 中的校验逻辑
- Contract testing for interface compliance / 接口合规性的契约测试
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """Fake vector store provider for testing. / 用于测试的 fake vector store provider。
    
    Maintains an in-memory dict to simulate upsert/query operations. / 维护内存 dict 来模拟 upsert/query 操作。
    """
    
    def __init__(self, settings: Any = None, **kwargs: Any):
        """Initialize fake vector store. / 初始化 fake vector store。
        
        Args: / 参数：
            settings: Optional settings (unused in fake). / 可选 settings（fake 中未使用）。
            **kwargs: Additional parameters (unused). / 额外参数（未使用）。
        """
        self.settings = settings
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.upsert_count = 0
        self.query_count = 0
    
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Store records in memory. / 将 records 存储到内存中。"""
        self.validate_records(records)
        self.upsert_count += 1
        
        for record in records:
            self.storage[record['id']] = record
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return stored records with fake similarity scores. / 返回带 fake similarity scores 的已存储 records。"""
        self.validate_query_vector(vector, top_k)
        self.query_count += 1
        
        # Simple fake: return all records with descending fake scores / 简单 fake：返回所有 records，并使用递减的 fake 分数
        results = []
        for i, (record_id, record) in enumerate(self.storage.items()):
            if i >= top_k:
                break
            
            # Apply metadata filters if provided / 如果提供 metadata filters，则应用过滤
            if filters:
                metadata = record.get('metadata', {})
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            results.append({
                'id': record_id,
                'score': 1.0 - (i * 0.1),  # Fake decreasing scores / fake 递减分数
                'metadata': record.get('metadata', {}),
                'vector': record.get('vector'),
            })
        
        return results


class TestBaseVectorStore:
    """Tests for BaseVectorStore abstract class. / BaseVectorStore 抽象类测试。"""
    
    def test_validate_records_success(self):
        """Valid records should pass validation. / 有效 records 应通过校验。"""
        store = FakeVectorStore()
        records = [
            {'id': 'doc1', 'vector': [0.1, 0.2, 0.3]},
            {'id': 'doc2', 'vector': [0.4, 0.5, 0.6], 'metadata': {'source': 'test.pdf'}},
        ]
        # Should not raise / 不应抛出异常
        store.validate_records(records)
    
    def test_validate_records_empty_list(self):
        """Empty records list should raise ValueError. / 空 records 列表应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="cannot be empty"):
            store.validate_records([])
    
    def test_validate_records_non_dict(self):
        """Non-dict records should raise ValueError. / 非 dict records 应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="not a dict"):
            store.validate_records([{'id': 'doc1', 'vector': [0.1]}, "invalid"])  # type: ignore
    
    def test_validate_records_missing_id(self):
        """Record missing 'id' field should raise ValueError. / record 缺失 'id' 字段应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="missing required field: 'id'"):
            store.validate_records([{'vector': [0.1, 0.2]}])
    
    def test_validate_records_missing_vector(self):
        """Record missing 'vector' field should raise ValueError. / record 缺失 'vector' 字段应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="missing required field: 'vector'"):
            store.validate_records([{'id': 'doc1'}])
    
    def test_validate_records_invalid_vector_type(self):
        """Vector with wrong type should raise ValueError. / vector 类型错误应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="invalid vector type"):
            store.validate_records([{'id': 'doc1', 'vector': "not a list"}])  # type: ignore
    
    def test_validate_records_empty_vector(self):
        """Empty vector should raise ValueError. / 空 vector 应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="empty vector"):
            store.validate_records([{'id': 'doc1', 'vector': []}])
    
    def test_validate_query_vector_success(self):
        """Valid query vector should pass validation. / 有效 query vector 应通过校验。"""
        store = FakeVectorStore()
        # Should not raise / 不应抛出异常
        store.validate_query_vector([0.1, 0.2, 0.3], top_k=10)
    
    def test_validate_query_vector_invalid_type(self):
        """Query vector with wrong type should raise ValueError. / query vector 类型错误应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="must be a list or tuple"):
            store.validate_query_vector("not a list", top_k=10)  # type: ignore
    
    def test_validate_query_vector_empty(self):
        """Empty query vector should raise ValueError. / 空 query vector 应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="cannot be empty"):
            store.validate_query_vector([], top_k=10)
    
    def test_validate_query_vector_invalid_top_k(self):
        """Invalid top_k should raise ValueError. / 无效 top_k 应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="must be a positive integer"):
            store.validate_query_vector([0.1, 0.2], top_k=0)
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            store.validate_query_vector([0.1, 0.2], top_k=-5)
    
    def test_delete_not_implemented(self):
        """delete() should raise NotImplementedError by default. / delete() 默认应抛出 NotImplementedError。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError, match="does not implement delete"):
            store.delete(['doc1'])
    
    def test_clear_not_implemented(self):
        """clear() should raise NotImplementedError by default. / clear() 默认应抛出 NotImplementedError。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError, match="does not implement clear"):
            store.clear()


class TestFakeVectorStore:
    """Tests for FakeVectorStore implementation. / FakeVectorStore 实现测试。"""
    
    def test_upsert_single_record(self):
        """Upserting single record should store it. / upsert 单条 record 应存储它。"""
        store = FakeVectorStore()
        records = [{'id': 'doc1', 'vector': [0.1, 0.2, 0.3]}]
        store.upsert(records)
        
        assert store.upsert_count == 1
        assert 'doc1' in store.storage
        assert store.storage['doc1']['vector'] == [0.1, 0.2, 0.3]
    
    def test_upsert_multiple_records(self):
        """Upserting multiple records should store them all. / upsert 多条 records 应全部存储。"""
        store = FakeVectorStore()
        records = [
            {'id': 'doc1', 'vector': [0.1, 0.2]},
            {'id': 'doc2', 'vector': [0.3, 0.4], 'metadata': {'source': 'test.pdf'}},
        ]
        store.upsert(records)
        
        assert len(store.storage) == 2
        assert 'doc1' in store.storage
        assert 'doc2' in store.storage
        assert store.storage['doc2']['metadata']['source'] == 'test.pdf'
    
    def test_upsert_idempotent(self):
        """Upserting same record multiple times should be idempotent. / 多次 upsert 同一 record 应保持幂等。"""
        store = FakeVectorStore()
        records = [{'id': 'doc1', 'vector': [0.1, 0.2]}]
        
        store.upsert(records)
        store.upsert(records)
        
        assert store.upsert_count == 2
        assert len(store.storage) == 1  # Still only one record / 仍然只有一条记录
    
    def test_upsert_validates_input(self):
        """upsert() should validate records and raise on invalid input. / upsert() 应校验 records 并在输入无效时抛出异常。"""
        store = FakeVectorStore()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            store.upsert([])
        
        with pytest.raises(ValueError, match="missing required field: 'id'"):
            store.upsert([{'vector': [0.1]}])
    
    def test_query_returns_results(self):
        """query() should return stored records with scores. / query() 应返回带分数的已存储 records。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': 'doc1', 'vector': [0.1, 0.2]},
            {'id': 'doc2', 'vector': [0.3, 0.4]},
        ])
        
        results = store.query(vector=[0.5, 0.6], top_k=10)
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc1'
        assert results[0]['score'] == 1.0
        assert results[1]['id'] == 'doc2'
        assert results[1]['score'] == 0.9
    
    def test_query_respects_top_k(self):
        """query() should limit results to top_k. / query() 应将结果限制为 top_k。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': f'doc{i}', 'vector': [float(i), float(i)]}
            for i in range(10)
        ])
        
        results = store.query(vector=[0.0, 0.0], top_k=3)
        
        assert len(results) == 3
    
    def test_query_with_filters(self):
        """query() should apply metadata filters. / query() 应应用 metadata filters。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': 'doc1', 'vector': [0.1], 'metadata': {'source': 'a.pdf'}},
            {'id': 'doc2', 'vector': [0.2], 'metadata': {'source': 'b.pdf'}},
            {'id': 'doc3', 'vector': [0.3], 'metadata': {'source': 'a.pdf'}},
        ])
        
        results = store.query(vector=[0.0], top_k=10, filters={'source': 'a.pdf'})
        
        # Should only return doc1 and doc3 / 应只返回 doc1 和 doc3
        result_ids = [r['id'] for r in results]
        assert 'doc1' in result_ids
        assert 'doc3' in result_ids
        assert 'doc2' not in result_ids
    
    def test_query_increments_count(self):
        """Each query should increment the counter. / 每次 query 都应递增计数器。"""
        store = FakeVectorStore()
        assert store.query_count == 0
        
        store.query(vector=[0.1], top_k=5)
        assert store.query_count == 1
        
        store.query(vector=[0.2], top_k=5)
        assert store.query_count == 2
    
    def test_query_validates_input(self):
        """query() should validate inputs and raise on invalid parameters. / query() 应校验输入并在参数无效时抛出异常。"""
        store = FakeVectorStore()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            store.query(vector=[], top_k=5)
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            store.query(vector=[0.1], top_k=0)


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory. / VectorStoreFactory 测试。"""
    
    def setup_method(self):
        """Reset factory registry before each test. / 每个测试前重置工厂注册表。"""
        VectorStoreFactory._PROVIDERS.clear()
    
    def test_register_provider_success(self):
        """Registering a valid provider should add it to registry. / 注册有效 provider 应将其加入注册表。"""
        VectorStoreFactory.register_provider('fake', FakeVectorStore)
        
        assert 'fake' in VectorStoreFactory._PROVIDERS
        assert VectorStoreFactory._PROVIDERS['fake'] is FakeVectorStore
    
    def test_register_provider_case_insensitive(self):
        """Provider names should be case-insensitive. / provider 名称应大小写不敏感。"""
        VectorStoreFactory.register_provider('FakeStore', FakeVectorStore)
        
        assert 'fakestore' in VectorStoreFactory._PROVIDERS
    
    def test_register_provider_invalid_class(self):
        """Registering non-BaseVectorStore class should raise ValueError. / 注册非 BaseVectorStore 类应抛出 ValueError。"""
        
        class NotAVectorStore:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseVectorStore"):
            VectorStoreFactory.register_provider('invalid', NotAVectorStore)  # type: ignore
    
    def test_create_with_registered_provider(self):
        """Creating instance with registered provider should work. / 使用已注册 provider 创建实例应正常工作。"""
        VectorStoreFactory.register_provider('fake', FakeVectorStore)
        
        # Mock settings / Mock settings
        settings = MagicMock()
        settings.vector_store.provider = 'fake'
        
        store = VectorStoreFactory.create(settings)
        
        assert isinstance(store, FakeVectorStore)
        assert store.settings is settings
    
    def test_create_case_insensitive_provider(self):
        """Provider lookup should be case-insensitive. / provider 查找应大小写不敏感。"""
        VectorStoreFactory.register_provider('fake', FakeVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'FAKE'
        
        store = VectorStoreFactory.create(settings)
        
        assert isinstance(store, FakeVectorStore)
    
    def test_create_unknown_provider(self):
        """Creating instance with unknown provider should raise ValueError. / 使用未知 provider 创建实例应抛出 ValueError。"""
        settings = MagicMock()
        settings.vector_store.provider = 'nonexistent'
        
        with pytest.raises(ValueError, match="Unsupported VectorStore provider: 'nonexistent'"):
            VectorStoreFactory.create(settings)
    
    def test_create_unknown_provider_shows_available(self):
        """Error message should list available providers. / 错误消息应列出可用 providers。"""
        VectorStoreFactory.register_provider('chroma', FakeVectorStore)
        VectorStoreFactory.register_provider('qdrant', FakeVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'unknown'
        
        with pytest.raises(ValueError, match="Available providers: chroma, qdrant"):
            VectorStoreFactory.create(settings)
    
    def test_create_missing_provider_field(self):
        """Missing provider field should raise ValueError with helpful message. / 缺失 provider 字段应抛出带有帮助信息的 ValueError。"""
        settings = MagicMock()
        del settings.vector_store  # Simulate missing config section / 模拟缺失 config section
        
        with pytest.raises(ValueError) as exc_info:
            VectorStoreFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Missing required configuration" in error_message
        assert "settings.vector_store.provider" in error_message
        assert "settings.yaml" in error_message
    
    def test_create_with_override_kwargs(self):
        """Factory should pass override kwargs to provider constructor. / 工厂应将覆盖 kwargs 传给 provider 构造函数。"""
        
        class ConfigurableVectorStore(BaseVectorStore):
            def __init__(self, settings: Any, custom_param: str = "default", **kwargs: Any):
                self.settings = settings
                self.custom_param = custom_param
            
            def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
                pass
            
            def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> List[Dict[str, Any]]:
                return []
        
        VectorStoreFactory.register_provider('configurable', ConfigurableVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'configurable'
        
        store = VectorStoreFactory.create(settings, custom_param="overridden")
        
        assert isinstance(store, ConfigurableVectorStore)
        assert store.custom_param == "overridden"
    
    def test_create_provider_instantiation_error(self):
        """Errors during provider instantiation should be wrapped with context. / provider 实例化过程中的错误应包装上下文。"""
        
        class FailingVectorStore(BaseVectorStore):
            def __init__(self, settings: Any, **kwargs: Any):
                raise RuntimeError("Simulated initialization failure")
            
            def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
                pass
            
            def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> List[Dict[str, Any]]:
                return []
        
        VectorStoreFactory.register_provider('failing', FailingVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'failing'
        
        with pytest.raises(RuntimeError, match="Failed to instantiate VectorStore provider 'failing'"):
            VectorStoreFactory.create(settings)
    
    def test_list_providers_empty(self):
        """list_providers() should return empty list when no providers registered. / 未注册 providers 时 list_providers() 应返回空列表。"""
        assert VectorStoreFactory.list_providers() == []
    
    def test_list_providers_with_providers(self):
        """list_providers() should return sorted list of provider names. / list_providers() 应返回排序后的 provider 名称列表。"""
        VectorStoreFactory.register_provider('chroma', FakeVectorStore)
        VectorStoreFactory.register_provider('qdrant', FakeVectorStore)
        VectorStoreFactory.register_provider('milvus', FakeVectorStore)
        
        providers = VectorStoreFactory.list_providers()
        
        assert providers == ['chroma', 'milvus', 'qdrant']  # Alphabetically sorted / 按字母排序


# ── Boundary / Contract tests (I4) ──────────────────────────────────

class TestBaseVectorStoreContractBoundary:
    """Boundary tests for BaseVectorStore contract. / BaseVectorStore 契约边界测试。"""

    def test_get_by_ids_not_implemented(self):
        """get_by_ids() should raise NotImplementedError by default. / get_by_ids() 默认应抛出 NotImplementedError。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError, match="does not implement get_by_ids"):
            store.get_by_ids(['doc1'])

    def test_validate_records_vector_non_numeric_accepted(self):
        """Vector containing non-numeric elements is accepted at base level. / 在 base 层接受包含非数字元素的 vector。
        
        Note: The base class only checks type (list/tuple) and non-empty. / 注意：base class 只检查类型（list/tuple）和非空。
        Type enforcement of individual elements is provider-specific. / 单个元素的类型约束由具体 provider 决定。
        """
        store = FakeVectorStore()
        # Base validation does not reject non-numeric elements / base 校验不会拒绝非数字元素
        store.validate_records([{'id': 'doc1', 'vector': ['a', 'b']}])

    def test_validate_records_single_element_vector(self):
        """Single-element vector should be valid. / 单元素 vector 应有效。"""
        store = FakeVectorStore()
        store.validate_records([{'id': 'doc1', 'vector': [0.5]}])

    def test_query_top_k_one(self):
        """top_k=1 should return at most 1 result. / top_k=1 应最多返回 1 条结果。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': 'a', 'vector': [0.1]},
            {'id': 'b', 'vector': [0.2]},
        ])
        results = store.query([0.1], top_k=1)
        assert len(results) <= 1

    def test_query_empty_store(self):
        """Querying empty store should return empty list. / 查询空 store 应返回空列表。"""
        store = FakeVectorStore()
        results = store.query([0.1, 0.2], top_k=5)
        assert results == []

    def test_upsert_preserves_metadata(self):
        """Metadata should be preserved through upsert → query cycle. / metadata 应在 upsert → query 周期中保留。"""
        store = FakeVectorStore()
        store.upsert([{
            'id': 'doc1',
            'vector': [0.1],
            'metadata': {'source': 'test.pdf', 'page': 3},
        }])
        results = store.query([0.1], top_k=1)
        assert results[0]['metadata']['source'] == 'test.pdf'
        assert results[0]['metadata']['page'] == 3

    def test_query_filters_empty_dict(self):
        """Empty filter dict should match all records. / 空 filter dict 应匹配所有 records。"""
        store = FakeVectorStore()
        store.upsert([{'id': 'a', 'vector': [0.1], 'metadata': {'k': 'v'}}])
        results = store.query([0.1], top_k=10, filters={})
        assert len(results) == 1

    def test_delete_requires_list(self):
        """delete() takes a list of IDs (contract shape check). / delete() 接收 ID 列表（契约形状检查）。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError):
            store.delete(['id1', 'id2'])
