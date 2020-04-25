from numpy import random
from sequence.protocols.management.memory_manager import MemoryInfo
from sequence.protocols.management.rule_manager import RuleManager, Rule

random.seed(1)


def test_Rule_do():
    class FakeRuleManager(RuleManager):
        def __init__(self):
            RuleManager.__init__(self)
            self.log = []

        def send_request(self, protocol, req_dst, req_condition):
            self.log.append((protocol, req_dst, req_condition))

    def fake_action(memories_info):
        if len(memories_info) == 1:
            return "protocol1", ["req_dst1"], ["req_condition1"]
        else:
            return "protocol2", [None], [None]

    rule_manager = FakeRuleManager()
    rule = Rule(1, fake_action, None)
    rule.set_rule_manager(rule_manager)
    assert rule.priority == 1 and len(rule.protocols) == 0
    memories_info = [MemoryInfo(None, 0)]
    rule.do(memories_info)
    assert len(rule.protocols) == 1 and rule.protocols[0] == "protocol1"
    assert len(rule_manager.log) == 1 and rule_manager.log[0] == ("protocol1", "req_dst1", "req_condition1")
    memories_info = [MemoryInfo(None, 0), MemoryInfo(None, 1)]
    rule.do(memories_info)
    assert len(rule.protocols) == 2 and rule.protocols[1] == "protocol2"
    assert len(rule_manager.log) == 2 and rule_manager.log[1] == ("protocol2", None, None)


def test_Rule_is_valid():
    class FakeRuleManager():
        def __init__(self):
            pass

        def get_memory_manager(self):
            return 0.5

    def fake_condition(val1, val2):
        return val1 < 0.5

    rule = Rule(1, None, fake_condition)
    rule.set_rule_manager(FakeRuleManager())
    for _ in range(100):
        val1 = random.random()
        assert rule.is_valid(val1) == (val1 < 0.5)


def test_RuleManager_load():
    rule_manager = RuleManager()
    for _ in range(100):
        priority = random.randint(20)
        rule = Rule(priority, None, None)
        rule_manager.load(rule)

    for i in range(1, len(rule_manager)):
        assert rule_manager[i].priority >= rule_manager[i - 1].priority
        assert id(rule_manager[i].rule_manager) == id(rule_manager)


def test_RuleManager_expire():
    ruleset = RuleManager()
    rule = Rule(1, None, None)
    rule.protocols.append("protocol")
    assert ruleset.load(rule) and len(ruleset) == 1
    protocol = ruleset.expire(rule)
    assert len(ruleset) == 0
    assert protocol == ["protocol"]
