from yggdrasill.task_nodes.roles import role_from_block_type, ALL_ROLES, KNOWN_ROLES


class TestRoleEnum:
    def test_all_seven_roles(self):
        assert len(ALL_ROLES) == 7
        names = {r.value for r in ALL_ROLES}
        assert names == {
            "backbone", "injector", "conjector",
            "inner_module", "outer_module",
            "helper", "converter",
        }


class TestKnownRoles:
    def test_known_roles_set(self):
        assert isinstance(KNOWN_ROLES, set)
        assert len(KNOWN_ROLES) == 7
        assert "inner_module" in KNOWN_ROLES


class TestRoleFromBlockType:
    def test_backbone(self):
        assert role_from_block_type("backbone/identity") == "backbone"

    def test_injector(self):
        assert role_from_block_type("injector/clip") == "injector"

    def test_converter(self):
        assert role_from_block_type("converter/vae_enc") == "converter"

    def test_unknown_prefix(self):
        assert role_from_block_type("unknown/foo") is None

    def test_no_slash(self):
        assert role_from_block_type("backbone") == "backbone"

    def test_case_insensitive(self):
        assert role_from_block_type("BACKBONE/Identity") == "backbone"

    def test_underscore_separator(self):
        assert role_from_block_type("backbone_unet2d") == "backbone"

    def test_inner_module_underscore(self):
        assert role_from_block_type("inner_module_ddim") == "inner_module"

    def test_inner_module_slash(self):
        assert role_from_block_type("inner_module/identity") == "inner_module"

    def test_ip_adapter_is_conjector(self):
        assert role_from_block_type("adapter/ip_adapter") == "conjector"

    def test_returns_string_not_enum(self):
        result = role_from_block_type("backbone")
        assert isinstance(result, str)
        assert result == "backbone"
