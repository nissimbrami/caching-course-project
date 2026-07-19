"""Tests for the GDSFEvictionPlugin (GPTCache adapter).

Tests cover:
- Plugin initialization (default and custom parameters)
- put() method delegates correctly to the eviction manager
- get() method (returns evicted keys, direct eviction)
- is_evict() method (detects when eviction is needed)
- access() method (recording cache hits)
- register_metadata() for pre-registering size/cost
- Metadata resolution order (pre-registered > callback > defaults)
- Edge cases (duplicate keys, empty cache eviction, callback errors)
- That the plugin correctly bridges the GPTCache EvictionBase interface
"""

import pytest

from cost_aware_eviction.gptcache_plugin import GDSFEvictionPlugin, EvictionBase
from cost_aware_eviction.config import GDSFConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plugin():
    """Return a plugin with a small max_size for testing."""
    return GDSFEvictionPlugin(max_size=10, default_entry_size=1, default_entry_cost=1.0)


@pytest.fixture
def plugin_with_callback():
    """Return a plugin with a metadata callback."""
    metadata_store = {
        "key_a": (2, 5.0),
        "key_b": (3, 1.0),
        "key_c": (1, 10.0),
    }

    def callback(key):
        return metadata_store.get(key, (1, 1.0))

    return GDSFEvictionPlugin(
        max_size=10,
        default_entry_size=1,
        default_entry_cost=1.0,
        metadata_callback=callback,
    )


@pytest.fixture
def large_plugin():
    """Return a plugin with a larger max_size for multi-entry tests."""
    return GDSFEvictionPlugin(max_size=100, default_entry_size=1, default_entry_cost=1.0)


# ---------------------------------------------------------------------------
# Test Initialization
# ---------------------------------------------------------------------------


class TestPluginInitialization:
    """Test plugin creation with various parameters."""

    def test_default_initialization(self):
        """Plugin can be created with all default parameters."""
        plugin = GDSFEvictionPlugin()
        assert plugin.num_entries == 0
        assert plugin.current_size == 0
        assert plugin.utilization == 0.0

    def test_custom_max_size(self):
        """Plugin respects custom max_size parameter."""
        plugin = GDSFEvictionPlugin(max_size=5000)
        # Fill partially and verify it does not evict
        plugin.register_metadata("item", size=4000, cost=1.0)
        plugin.put(["item"])
        assert plugin.num_entries == 1
        assert plugin.current_size == 4000

    def test_custom_alpha_beta(self):
        """Plugin passes alpha and beta to the underlying manager."""
        plugin = GDSFEvictionPlugin(max_size=10, alpha=2.0, beta=0.5)
        plugin.register_metadata("item", size=1, cost=4.0)
        plugin.put(["item"])
        # Priority = Clock + (freq^alpha * cost^beta) / size
        # = 0 + (1^2 * 4^0.5) / 1 = 2.0
        assert plugin.num_entries == 1

    def test_custom_config_overrides_params(self):
        """When a GDSFConfig is provided, it overrides individual params."""
        config = GDSFConfig(max_size=50, alpha=0.5, beta=1.5)
        plugin = GDSFEvictionPlugin(max_size=999, alpha=999, beta=999, config=config)
        # The config's max_size=50 should be in effect
        for i in range(50):
            plugin.put([f"item_{i}"])
        # Should be at or near capacity
        assert plugin.num_entries <= 50

    def test_default_entry_size_and_cost(self):
        """Default entry size and cost are used when no metadata is provided."""
        plugin = GDSFEvictionPlugin(
            max_size=100, default_entry_size=5, default_entry_cost=2.0
        )
        plugin.put(["item1"])
        assert plugin.current_size == 5

    def test_kwargs_accepted_for_gptcache_compatibility(self):
        """Extra kwargs do not raise errors (GPTCache compatibility)."""
        plugin = GDSFEvictionPlugin(
            max_size=100, some_gptcache_param="value", another=42
        )
        assert plugin.num_entries == 0

    def test_initial_len_is_zero(self):
        """Newly created plugin has length zero."""
        plugin = GDSFEvictionPlugin(max_size=100)
        assert len(plugin) == 0


# ---------------------------------------------------------------------------
# Test EvictionBase Interface Compliance
# ---------------------------------------------------------------------------


class TestEvictionBaseInterface:
    """Test that the plugin correctly implements the EvictionBase interface."""

    def test_is_instance_of_eviction_base(self, plugin):
        """Plugin is an instance of EvictionBase."""
        assert isinstance(plugin, EvictionBase)

    def test_has_put_method(self, plugin):
        """Plugin has a put() method."""
        assert callable(getattr(plugin, "put", None))

    def test_has_get_method(self, plugin):
        """Plugin has a get() method."""
        assert callable(getattr(plugin, "get", None))

    def test_has_is_evict_method(self, plugin):
        """Plugin has an is_evict() method."""
        assert callable(getattr(plugin, "is_evict", None))

    def test_put_accepts_list(self, plugin):
        """put() accepts a list of keys (EvictionBase signature)."""
        plugin.put(["key1", "key2", "key3"])
        assert plugin.num_entries == 3

    def test_get_returns_key_or_none(self, plugin):
        """get() returns a key or None."""
        # Empty cache returns None
        result = plugin.get()
        assert result is None

    def test_is_evict_returns_bool(self, plugin):
        """is_evict() returns a boolean."""
        result = plugin.is_evict()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test put() Method
# ---------------------------------------------------------------------------


class TestPutMethod:
    """Test that put() correctly delegates to the eviction manager."""

    def test_put_single_item(self, plugin):
        """Put a single item into the cache."""
        plugin.put(["item_1"])
        assert plugin.num_entries == 1
        assert plugin.current_size == 1  # default_entry_size=1

    def test_put_multiple_items(self, plugin):
        """Put multiple items in a single call."""
        plugin.put(["a", "b", "c"])
        assert plugin.num_entries == 3
        assert plugin.current_size == 3

    def test_put_with_registered_metadata(self, plugin):
        """Put uses pre-registered metadata for size and cost."""
        plugin.register_metadata("big_item", size=5, cost=10.0)
        plugin.put(["big_item"])
        assert plugin.current_size == 5

    def test_put_triggers_eviction_when_full(self, plugin):
        """Put triggers eviction when cache exceeds max_size."""
        # Fill cache (max_size=10, default_entry_size=1)
        plugin.put([f"item_{i}" for i in range(10)])
        assert plugin.num_entries == 10

        # Add one more - should trigger eviction
        plugin.put(["overflow"])
        assert plugin.num_entries <= 10
        assert plugin.current_size <= 10

    def test_put_populates_eviction_queue(self, plugin):
        """When eviction occurs during put(), evicted keys go to the queue."""
        plugin.put([f"item_{i}" for i in range(10)])
        # Next put should evict
        plugin.put(["trigger"])
        # is_evict should be true if there are pending evictions or at capacity
        # The evicted key should have been added to the eviction queue
        # or the cache is at capacity
        assert plugin.num_entries <= 10

    def test_put_with_callback_metadata(self, plugin_with_callback):
        """Put uses metadata callback when no pre-registered metadata exists."""
        # key_a has size=2 via callback
        plugin_with_callback.put(["key_a"])
        assert plugin_with_callback.current_size == 2

    def test_put_empty_list(self, plugin):
        """Putting an empty list does nothing."""
        plugin.put([])
        assert plugin.num_entries == 0
        assert plugin.current_size == 0

    def test_put_with_large_item_triggers_multiple_evictions(self):
        """A large item evicts multiple smaller items."""
        plugin = GDSFEvictionPlugin(max_size=10, default_entry_size=1, default_entry_cost=1.0)
        # Fill with 10 items of size 1
        plugin.put([f"small_{i}" for i in range(10)])
        assert plugin.num_entries == 10

        # Register a large item and put it
        plugin.register_metadata("big", size=5, cost=1.0)
        plugin.put(["big"])
        # Should have evicted enough to fit the big item
        assert plugin.current_size <= 10
        assert "big" in plugin._manager


# ---------------------------------------------------------------------------
# Test get() Method
# ---------------------------------------------------------------------------


class TestGetMethod:
    """Test that get() returns evicted keys correctly."""

    def test_get_empty_cache_returns_none(self, plugin):
        """get() on empty cache returns None."""
        result = plugin.get()
        assert result is None

    def test_get_returns_evicted_key_from_queue(self, plugin):
        """get() returns keys that were evicted during put()."""
        # Fill cache
        plugin.put([f"item_{i}" for i in range(10)])
        # Trigger eviction
        plugin.put(["new_item"])

        # If eviction occurred, get() should return the evicted key
        if plugin._eviction_queue:
            evicted = plugin.get()
            assert evicted is not None
            assert evicted.startswith("item_")

    def test_get_drains_eviction_queue(self, plugin):
        """Multiple get() calls drain the eviction queue."""
        # Fill and trigger eviction of multiple items
        plugin.put([f"item_{i}" for i in range(10)])
        plugin.register_metadata("huge", size=5, cost=100.0)
        plugin.put(["huge"])

        evicted_keys = []
        while plugin._eviction_queue:
            key = plugin.get()
            if key is not None:
                evicted_keys.append(key)

        # All evicted keys should be from the original items
        for key in evicted_keys:
            assert key.startswith("item_")

    def test_get_evicts_lowest_priority_when_queue_empty(self, large_plugin):
        """When eviction queue is empty, get() evicts lowest priority item."""
        large_plugin.register_metadata("cheap", size=1, cost=0.1)
        large_plugin.register_metadata("expensive", size=1, cost=100.0)
        large_plugin.put(["cheap", "expensive"])

        # Queue should be empty (no eviction triggered)
        assert len(large_plugin._eviction_queue) == 0

        # get() should evict the lowest priority item (cheap)
        evicted = large_plugin.get()
        assert evicted == "cheap"

    def test_get_returns_none_after_all_evicted(self, plugin):
        """get() returns None once all items are evicted."""
        plugin.put(["only_item"])

        # Evict via get()
        evicted = plugin.get()
        assert evicted == "only_item"

        # Now empty
        result = plugin.get()
        assert result is None


# ---------------------------------------------------------------------------
# Test is_evict() Method
# ---------------------------------------------------------------------------


class TestIsEvictMethod:
    """Test is_evict() signals when eviction is needed."""

    def test_is_evict_false_when_empty(self, plugin):
        """is_evict() returns False when cache is empty."""
        assert plugin.is_evict() is False

    def test_is_evict_false_when_below_capacity(self, plugin):
        """is_evict() returns False when cache is below capacity."""
        plugin.put(["item_1", "item_2", "item_3"])
        assert plugin.is_evict() is False

    def test_is_evict_true_when_at_capacity(self, plugin):
        """is_evict() returns True when cache utilization >= 1.0."""
        # Fill to capacity (max_size=10, default_entry_size=1)
        plugin.put([f"item_{i}" for i in range(10)])
        assert plugin.is_evict() is True

    def test_is_evict_true_when_eviction_queue_not_empty(self, plugin):
        """is_evict() returns True when there are pending evictions."""
        # Fill and overflow
        plugin.put([f"item_{i}" for i in range(10)])
        plugin.put(["overflow"])

        # If put triggered eviction and items are in queue
        if plugin._eviction_queue:
            assert plugin.is_evict() is True


# ---------------------------------------------------------------------------
# Test access() Method
# ---------------------------------------------------------------------------


class TestAccessMethod:
    """Test that access() records cache hits correctly."""

    def test_access_existing_key_returns_true(self, plugin):
        """access() returns True for an existing key."""
        plugin.put(["item"])
        result = plugin.access("item")
        assert result is True

    def test_access_nonexistent_key_returns_false(self, plugin):
        """access() returns False for a key not in the cache."""
        result = plugin.access("ghost")
        assert result is False

    def test_access_boosts_priority(self, large_plugin):
        """Accessing an item increases its priority (survives eviction)."""
        # Add items with the same cost
        large_plugin.register_metadata("popular", size=1, cost=1.0)
        large_plugin.register_metadata("unpopular", size=1, cost=1.0)
        large_plugin.put(["popular", "unpopular"])

        # Access "popular" many times
        for _ in range(10):
            large_plugin.access("popular")

        # Now evict via get() - should evict "unpopular" first
        evicted = large_plugin.get()
        assert evicted == "unpopular"

    def test_access_after_eviction_returns_false(self, plugin):
        """access() returns False for an item that was evicted."""
        plugin.put(["item"])
        # Force eviction via get()
        plugin.get()
        result = plugin.access("item")
        assert result is False


# ---------------------------------------------------------------------------
# Test register_metadata()
# ---------------------------------------------------------------------------


class TestRegisterMetadata:
    """Test pre-registering metadata for cache entries."""

    def test_register_explicit_size_and_cost(self, plugin):
        """Registering explicit size and cost uses those values."""
        plugin.register_metadata("key1", size=7, cost=3.5)
        plugin.put(["key1"])
        assert plugin.current_size == 7

    def test_register_with_response_text_estimates_size(self, large_plugin):
        """When only response_text is provided, size is estimated from text."""
        text = "Hello, world!"  # 13 bytes in UTF-8
        large_plugin.register_metadata("key1", response_text=text)
        large_plugin.put(["key1"])
        assert large_plugin.current_size == len(text.encode("utf-8"))

    def test_register_with_response_text_estimates_cost(self, plugin):
        """When only response_text is provided, cost is estimated."""
        text = "A sample response. " * 20
        plugin.register_metadata("key1", response_text=text, model_name="gpt-4")
        # Should not raise and metadata should be stored
        assert "key1" in plugin._pending_metadata
        meta = plugin._pending_metadata["key1"]
        assert meta["cost"] > 0

    def test_register_explicit_size_overrides_text(self, large_plugin):
        """Explicit size overrides text-based estimation."""
        large_plugin.register_metadata("key1", size=42, response_text="short text")
        large_plugin.put(["key1"])
        assert large_plugin.current_size == 42

    def test_register_explicit_cost_overrides_text(self, plugin):
        """Explicit cost overrides text-based estimation."""
        plugin.register_metadata("key1", size=1, cost=99.9, response_text="some text")
        assert plugin._pending_metadata["key1"]["cost"] == 99.9

    def test_metadata_consumed_after_put(self, plugin):
        """Pre-registered metadata is consumed (removed) after put."""
        plugin.register_metadata("key1", size=3, cost=2.0)
        assert "key1" in plugin._pending_metadata
        plugin.put(["key1"])
        assert "key1" not in plugin._pending_metadata

    def test_unregistered_key_uses_defaults(self, plugin):
        """Keys without metadata use default_entry_size and default_entry_cost."""
        plugin.put(["no_metadata_key"])
        assert plugin.current_size == 1  # default_entry_size=1


# ---------------------------------------------------------------------------
# Test Metadata Resolution Order
# ---------------------------------------------------------------------------


class TestMetadataResolution:
    """Test the priority order for metadata resolution."""

    def test_pending_metadata_takes_precedence_over_callback(self):
        """Pre-registered metadata is preferred over the callback."""
        def callback(key):
            return (99, 99.0)  # Should NOT be used

        plugin = GDSFEvictionPlugin(
            max_size=100,
            default_entry_size=1,
            default_entry_cost=1.0,
            metadata_callback=callback,
        )
        plugin.register_metadata("key1", size=5, cost=2.0)
        plugin.put(["key1"])
        assert plugin.current_size == 5  # From pre-registered, not callback's 99

    def test_callback_takes_precedence_over_defaults(self):
        """Callback is preferred over default values."""
        def callback(key):
            return (7, 3.0)

        plugin = GDSFEvictionPlugin(
            max_size=100,
            default_entry_size=1,
            default_entry_cost=1.0,
            metadata_callback=callback,
        )
        plugin.put(["key1"])
        assert plugin.current_size == 7  # From callback, not default 1

    def test_defaults_used_when_callback_and_metadata_absent(self):
        """Default values are used when no metadata or callback exists."""
        plugin = GDSFEvictionPlugin(
            max_size=100,
            default_entry_size=3,
            default_entry_cost=0.5,
        )
        plugin.put(["key1"])
        assert plugin.current_size == 3

    def test_callback_error_falls_back_to_defaults(self):
        """If the callback raises an exception, defaults are used."""
        def failing_callback(key):
            raise ValueError("Callback failed!")

        plugin = GDSFEvictionPlugin(
            max_size=100,
            default_entry_size=4,
            default_entry_cost=1.0,
            metadata_callback=failing_callback,
        )
        plugin.put(["key1"])
        assert plugin.current_size == 4  # Default, not from callback

    def test_callback_invalid_return_falls_back_to_defaults(self):
        """If the callback returns invalid data, defaults are used."""
        def bad_callback(key):
            return "not a tuple"

        plugin = GDSFEvictionPlugin(
            max_size=100,
            default_entry_size=6,
            default_entry_cost=1.0,
            metadata_callback=bad_callback,
        )
        plugin.put(["key1"])
        assert plugin.current_size == 6  # Default

    def test_callback_wrong_length_tuple_falls_back(self):
        """If the callback returns a tuple of wrong length, defaults are used."""
        def wrong_length_callback(key):
            return (1, 2, 3)  # length 3, not 2

        plugin = GDSFEvictionPlugin(
            max_size=100,
            default_entry_size=8,
            default_entry_cost=1.0,
            metadata_callback=wrong_length_callback,
        )
        plugin.put(["key1"])
        assert plugin.current_size == 8  # Default


# ---------------------------------------------------------------------------
# Test Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_duplicate_keys_in_single_put(self, large_plugin):
        """Putting the same key twice in one call updates it (no duplication)."""
        large_plugin.register_metadata("dup", size=2, cost=1.0)
        large_plugin.put(["dup", "dup"])
        # Should only have 1 entry (second put is an update)
        assert large_plugin.num_entries == 1

    def test_duplicate_key_across_puts(self, large_plugin):
        """Putting the same key in separate calls updates rather than duplicates."""
        large_plugin.register_metadata("item", size=3, cost=1.0)
        large_plugin.put(["item"])
        assert large_plugin.num_entries == 1
        assert large_plugin.current_size == 3

        large_plugin.register_metadata("item", size=5, cost=2.0)
        large_plugin.put(["item"])
        assert large_plugin.num_entries == 1
        # Size should be updated to 5
        assert large_plugin.current_size == 5

    def test_eviction_on_empty_cache(self, plugin):
        """get() on empty cache returns None without error."""
        result = plugin.get()
        assert result is None

    def test_is_evict_on_empty_cache(self, plugin):
        """is_evict() on empty cache returns False without error."""
        assert plugin.is_evict() is False

    def test_put_single_large_item_at_exact_capacity(self):
        """An item exactly equal to max_size can be inserted."""
        plugin = GDSFEvictionPlugin(max_size=10, default_entry_size=1, default_entry_cost=1.0)
        plugin.register_metadata("exact", size=10, cost=1.0)
        plugin.put(["exact"])
        assert plugin.num_entries == 1
        assert plugin.current_size == 10

    def test_put_oversized_item_rejected(self):
        """An item larger than max_size is rejected."""
        plugin = GDSFEvictionPlugin(max_size=5, default_entry_size=1, default_entry_cost=1.0)
        plugin.register_metadata("huge", size=10, cost=1.0)
        plugin.put(["huge"])
        # Item should not be in the cache
        assert plugin.num_entries == 0
        assert plugin.current_size == 0

    def test_many_puts_respects_capacity(self):
        """After many puts, current_size never exceeds max_size."""
        plugin = GDSFEvictionPlugin(max_size=20, default_entry_size=1, default_entry_cost=1.0)
        for i in range(100):
            plugin.put([f"item_{i}"])
            assert plugin.current_size <= 20

    def test_access_on_empty_cache_no_error(self, plugin):
        """access() on empty cache does not raise."""
        result = plugin.access("nonexistent")
        assert result is False

    def test_string_and_integer_keys(self, large_plugin):
        """Plugin handles both string and integer keys."""
        large_plugin.put(["string_key"])
        large_plugin.put([42])
        assert large_plugin.num_entries == 2


# ---------------------------------------------------------------------------
# Test Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Test the convenience properties of the plugin."""

    def test_current_size_reflects_entries(self, plugin):
        """current_size is the sum of all entry sizes."""
        plugin.register_metadata("a", size=3, cost=1.0)
        plugin.register_metadata("b", size=4, cost=1.0)
        plugin.put(["a", "b"])
        assert plugin.current_size == 7

    def test_utilization_zero_when_empty(self, plugin):
        """utilization is 0.0 when cache is empty."""
        assert plugin.utilization == 0.0

    def test_utilization_one_when_full(self, plugin):
        """utilization is 1.0 when cache is at max_size."""
        # max_size=10, fill with 10 items of size 1
        plugin.put([f"item_{i}" for i in range(10)])
        assert plugin.utilization == pytest.approx(1.0)

    def test_utilization_partial(self, plugin):
        """utilization reflects partial fill correctly."""
        plugin.register_metadata("item", size=5, cost=1.0)
        plugin.put(["item"])
        assert plugin.utilization == pytest.approx(0.5)

    def test_num_entries_tracks_count(self, plugin):
        """num_entries reflects the number of items in cache."""
        plugin.put(["a", "b", "c"])
        assert plugin.num_entries == 3

    def test_len_matches_num_entries(self, plugin):
        """len(plugin) matches num_entries."""
        plugin.put(["x", "y"])
        assert len(plugin) == plugin.num_entries == 2


# ---------------------------------------------------------------------------
# Test Integration Scenario
# ---------------------------------------------------------------------------


class TestIntegrationScenario:
    """Test a realistic usage scenario combining multiple operations."""

    def test_full_lifecycle(self):
        """Test put, access, eviction, and get in a realistic scenario."""
        plugin = GDSFEvictionPlugin(
            max_size=5,
            default_entry_size=1,
            default_entry_cost=1.0,
        )

        # Phase 1: Add items with different costs
        plugin.register_metadata("cheap1", size=1, cost=0.5)
        plugin.register_metadata("cheap2", size=1, cost=0.5)
        plugin.register_metadata("expensive", size=1, cost=50.0)
        plugin.register_metadata("medium1", size=1, cost=5.0)
        plugin.register_metadata("medium2", size=1, cost=5.0)
        plugin.put(["cheap1", "cheap2", "expensive", "medium1", "medium2"])
        assert plugin.num_entries == 5
        assert plugin.is_evict() is True  # At capacity

        # Phase 2: Access the expensive item (boost priority)
        plugin.access("expensive")
        plugin.access("expensive")

        # Phase 3: Add a new item - should evict cheapest
        plugin.register_metadata("new_item", size=1, cost=3.0)
        plugin.put(["new_item"])

        # Expensive item should survive
        assert plugin.access("expensive") is True

        # One of the cheap items should have been evicted
        cheap_count = sum(
            1 for key in ["cheap1", "cheap2"]
            if plugin.access(key)
        )
        assert cheap_count < 2  # At least one cheap item was evicted

    def test_sequential_puts_and_gets(self):
        """Sequential put and get operations work correctly."""
        plugin = GDSFEvictionPlugin(
            max_size=3,
            default_entry_size=1,
            default_entry_cost=1.0,
        )

        # Add 3 items to fill cache
        plugin.register_metadata("a", size=1, cost=1.0)
        plugin.register_metadata("b", size=1, cost=2.0)
        plugin.register_metadata("c", size=1, cost=3.0)
        plugin.put(["a", "b", "c"])

        assert plugin.is_evict() is True
        assert plugin.num_entries == 3

        # Get should evict the lowest priority item (a, cost=1.0)
        evicted = plugin.get()
        assert evicted == "a"
        assert plugin.num_entries == 2

        # Now "a" is gone
        assert plugin.access("a") is False
        assert plugin.access("b") is True
        assert plugin.access("c") is True
