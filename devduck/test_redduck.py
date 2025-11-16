"""🦆 DevDuck test suite"""


def test_import():
    """Test basic import and initialization"""
    try:
        import devduck

        print("✅ Import successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_status():
    """Test status function"""
    try:
        import devduck

        status = devduck.status()
        print(f"✅ Status: {status}")
        return True
    except Exception as e:
        print(f"❌ Status failed: {e}")
        return False


def test_basic_query():
    """Test basic agent query"""
    try:
        import devduck

        result = devduck.ask("what's 2+2?")
        print(f"✅ Query result: {result}")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False


def test_time_query():
    """Test current time tool"""
    try:
        import devduck

        result = devduck.ask("what time is it?")
        print(f"✅ Time query: {result}")
        return True
    except Exception as e:
        print(f"❌ Time query failed: {e}")
        return False


def run_tests():
    """Run all tests"""
    print("🦆 Testing Devduck...")

    tests = [test_import, test_status, test_basic_query, test_time_query]

    results = []
    for test in tests:
        print(f"\n🧪 Running {test.__name__}...")
        results.append(test())

    passed = sum(results)
    total = len(results)

    print(f"\n🦆 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Devduck is ready to go!")
    else:
        print("⚠️  Some tests failed. Check ollama service and dependencies.")


if __name__ == "__main__":
    run_tests()
