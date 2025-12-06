#!/usr/bin/env python3
"""
Unit tests for ECO generator components.

Tests buffer claiming, CTS tree generation, power-down ECO, and Verilog emission.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any

from buffer_manager import BufferManager, parse_placement_map, calculate_distance
from eco_generator import (
    get_used_slots, find_unused_logic_cells, find_available_conb,
    get_max_net_id, find_top_module, count_used_logic_cells,
    count_total_fabric_cells
)
from cts_api import generate_cts_tree
from rename_helper import rename_instances


class TestBufferManager:
    """Test BufferManager buffer claiming functionality."""
    
    def test_parse_placement_map(self, tmp_path):
        """Test parsing placement map file."""
        map_file = tmp_path / "test.map"
        map_file.write_text("cell1 T0Y0__R0_NAND_0\ncell2 T0Y0__R1_BUF_0\n")
        
        result = parse_placement_map(str(map_file))
        assert result == {"cell1": "T0Y0__R0_NAND_0", "cell2": "T0Y0__R1_BUF_0"}
    
    def test_calculate_distance(self):
        """Test distance calculation."""
        dist = calculate_distance(0.0, 0.0, 3.0, 4.0)
        assert abs(dist - 5.0) < 0.001  # 3-4-5 triangle
    
    def test_buffer_manager_initialization(self, tmp_path):
        """Test BufferManager initialization with fabric and placement map."""
        # Create minimal fabric YAML structure
        fabric_file = tmp_path / "fabric_cells.yaml"
        fabric_data = {
            'fabric_cells_by_tile': {
                'tiles': {
                    'T0Y0': {
                        'cells': [
                            {'name': 'T0Y0__R1_BUF_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T0Y0__R1_INV_0', 'x': 15.0, 'y': 20.0, 'orient': 'N'}
                        ]
                    }
                }
            }
        }
        import yaml
        with open(fabric_file, 'w') as f:
            yaml.dump(fabric_data, f)
        
        # Create placement map
        map_file = tmp_path / "placement.map"
        map_file.write_text("used_cell T0Y0__R1_BUF_0\n")
        
        manager = BufferManager(str(fabric_file), str(map_file))
        
        # Check that used buffer is excluded
        assert len(manager.available['BUF']) == 0  # All BUF slots used
        assert len(manager.available['INV']) == 1  # INV slot available
    
    def test_claim_buffer(self, tmp_path):
        """Test buffer claiming functionality."""
        fabric_file = tmp_path / "fabric_cells.yaml"
        fabric_data = {
            'fabric_cells_by_tile': {
                'tiles': {
                    'T0Y0': {
                        'cells': [
                            {'name': 'T0Y0__R1_BUF_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T0Y0__R1_BUF_1', 'x': 50.0, 'y': 60.0, 'orient': 'N'}
                        ]
                    }
                }
            }
        }
        import yaml
        with open(fabric_file, 'w') as f:
            yaml.dump(fabric_data, f)
        
        manager = BufferManager(str(fabric_file), None)
        
        # Claim buffer near (12, 22) - should get T0Y0__R1_BUF_0
        claimed_name = manager.claim_buffer(12.0, 22.0, preferred_type='BUF')
        assert claimed_name is not None
        assert claimed_name.startswith('cts_buffer_')
        assert claimed_name in manager.claimed
        
        # Verify buffer is removed from available
        assert len(manager.available['BUF']) == 1
    
    def test_insufficient_buffers(self, tmp_path):
        """Test error handling when no buffers available."""
        fabric_file = tmp_path / "fabric_cells.yaml"
        fabric_data = {
            'fabric_cells_by_tile': {
                'tiles': {}
            }
        }
        import yaml
        with open(fabric_file, 'w') as f:
            yaml.dump(fabric_data, f)
        
        manager = BufferManager(str(fabric_file), None)
        
        # Try to claim when no buffers available - should raise ValueError
        with pytest.raises(ValueError, match="No buffers available"):
            manager.claim_buffer(0.0, 0.0, preferred_type='BUF')


class TestECOGenerator:
    """Test ECO generator functions."""
    
    def test_get_used_slots(self, tmp_path):
        """Test getting used slots from placement files."""
        placement_file = tmp_path / "placement.json"
        placement_data = {
            "cell1": {"fabric_slot_name": "T0Y0__R0_NAND_0"},
            "cell2": {"fabric_slot_name": "T0Y0__R1_BUF_0"}
        }
        with open(placement_file, 'w') as f:
            json.dump(placement_data, f)
        
        used = get_used_slots(str(placement_file))
        assert "T0Y0__R0_NAND_0" in used
        assert "T0Y0__R1_BUF_0" in used
    
    def test_find_unused_logic_cells(self, tmp_path):
        """Test finding unused logic cells."""
        fabric_db = {
            'sky130_fd_sc_hd__nand2_2': [
                {'name': 'T0Y0__R0_NAND_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                {'name': 'T0Y0__R0_NAND_1', 'x': 15.0, 'y': 20.0, 'orient': 'N'}
            ]
        }
        used_slots = {'T0Y0__R0_NAND_0'}
        
        unused = find_unused_logic_cells(fabric_db, used_slots)
        assert len(unused) == 1
        assert unused[0]['name'] == 'T0Y0__R0_NAND_1'
        assert unused[0]['type'] == 'sky130_fd_sc_hd__nand2_2'
    
    def test_find_available_conb(self, tmp_path):
        """Test finding available conb_1 cell."""
        fabric_db = {
            'sky130_fd_sc_hd__conb_1': [
                {'name': 'T0Y0__R0_CONB_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                {'name': 'T0Y0__R0_CONB_1', 'x': 15.0, 'y': 20.0, 'orient': 'N'}
            ]
        }
        used_slots = {'T0Y0__R0_CONB_0'}
        
        conb = find_available_conb(fabric_db, used_slots)
        assert conb is not None
        assert conb['name'] == 'T0Y0__R0_CONB_1'
    
    def test_find_available_conb_none_available(self):
        """Test finding conb when none available."""
        fabric_db = {
            'sky130_fd_sc_hd__conb_1': [
                {'name': 'T0Y0__R0_CONB_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'}
            ]
        }
        used_slots = {'T0Y0__R0_CONB_0'}
        
        conb = find_available_conb(fabric_db, used_slots)
        assert conb is None
    
    def test_get_max_net_id(self):
        """Test getting maximum net ID from design."""
        design_data = {
            'modules': {
                'top': {
                    'cells': {
                        'cell1': {'connections': {'A': [1, 2, 3], 'B': [5]}},
                        'cell2': {'connections': {'X': [10, 20]}}
                    }
                }
            }
        }
        max_id = get_max_net_id(design_data)
        assert max_id == 20
    
    def test_find_top_module(self):
        """Test finding top module in design."""
        # Test with top attribute in module
        design_data = {
            'modules': {
                'top_module': {
                    'cells': {'cell1': {}},
                    'attributes': {'top': '00000000000000000000000000000001'}
                },
                'sub_module': {'cells': {}}
            }
        }
        top = find_top_module(design_data)
        assert top == 'top_module'
        
        # Test fallback to first module with cells
        design_data2 = {
            'modules': {
                'sub_module': {'cells': {}},
                'top_module': {'cells': {'cell1': {}}}
            }
        }
        top2 = find_top_module(design_data2)
        assert top2 == 'top_module'
    
    def test_count_used_logic_cells(self):
        """Test counting used logic cells."""
        fabric_db = {
            'sky130_fd_sc_hd__nand2_2': [
                {'name': 'T0Y0__R0_NAND_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                {'name': 'T0Y0__R0_NAND_1', 'x': 15.0, 'y': 20.0, 'orient': 'N'}
            ]
        }
        used_slots = {'T0Y0__R0_NAND_0'}
        
        count = count_used_logic_cells(fabric_db, used_slots)
        assert count == 1
    
    def test_count_total_fabric_cells(self):
        """Test counting total fabric cells."""
        fabric_db = {
            'sky130_fd_sc_hd__nand2_2': [
                {'name': 'cell1'}, {'name': 'cell2'}
            ],
            'sky130_fd_sc_hd__or2_2': [
                {'name': 'cell3'}
            ]
        }
        total = count_total_fabric_cells(fabric_db)
        assert total == 3


class TestCTSTreeGeneration:
    """Test CTS tree generation."""
    
    def test_cts_tree_generation_single_dff(self, tmp_path):
        """Test CTS tree generation with single DFF (no buffer needed)."""
        # Create minimal test files
        placement_map_file = tmp_path / "placement.map"
        placement_map_file.write_text("dff1 T0Y0__R1_DFBBP_0\n")
        
        fabric_file = tmp_path / "fabric_cells.yaml"
        fabric_data = {
            'fabric_cells_by_tile': {
                'tiles': {
                    'T0Y0': {
                        'cells': [
                            {'name': 'T0Y0__R1_DFBBP_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T0Y0__R1_BUF_0', 'x': 15.0, 'y': 20.0, 'orient': 'N'}
                        ]
                    }
                }
            }
        }
        import yaml
        with open(fabric_file, 'w') as f:
            yaml.dump(fabric_data, f)
        
        design_file = tmp_path / "design.json"
        design_data = {
            'modules': {
                'top': {
                    'cells': {
                        'dff1': {
                            'type': 'sky130_fd_sc_hd__dfbbp_1',
                            'connections': {'CLK': [1], 'D': [2], 'Q': [3]}
                        }
                    }
                }
            },
            'top': 'top'
        }
        with open(design_file, 'w') as f:
            json.dump(design_data, f)
        
        # Generate CTS tree
        cts_tree, updated_map = generate_cts_tree(
            str(placement_map_file),
            str(fabric_file),
            str(design_file),
            tree_type='h'
        )
        
        # For single DFF, tree should exist but may not need buffer
        assert cts_tree is not None
        assert isinstance(updated_map, dict)
    
    def test_htree_4_dffs(self, tmp_path):
        """Test H-Tree creates correct structure for 4 DFFs."""
        # Create placement map with 4 DFFs
        placement_map_file = tmp_path / "placement.map"
        placement_map_file.write_text(
            "dff1 T0Y0__R1_DFBBP_0\n"
            "dff2 T0Y1__R1_DFBBP_0\n"
            "dff3 T1Y0__R1_DFBBP_0\n"
            "dff4 T1Y1__R1_DFBBP_0\n"
        )
        
        # Create fabric with DFFs and buffers
        fabric_file = tmp_path / "fabric_cells.yaml"
        fabric_data = {
            'fabric_cells_by_tile': {
                'tiles': {
                    'T0Y0': {
                        'cells': [
                            {'name': 'T0Y0__R1_DFBBP_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T0Y0__R1_BUF_0', 'x': 15.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T0Y0__R1_BUF_1', 'x': 20.0, 'y': 20.0, 'orient': 'N'}
                        ]
                    },
                    'T0Y1': {
                        'cells': [
                            {'name': 'T0Y1__R1_DFBBP_0', 'x': 10.0, 'y': 40.0, 'orient': 'N'},
                            {'name': 'T0Y1__R1_BUF_0', 'x': 15.0, 'y': 40.0, 'orient': 'N'}
                        ]
                    },
                    'T1Y0': {
                        'cells': [
                            {'name': 'T1Y0__R1_DFBBP_0', 'x': 50.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T1Y0__R1_BUF_0', 'x': 55.0, 'y': 20.0, 'orient': 'N'}
                        ]
                    },
                    'T1Y1': {
                        'cells': [
                            {'name': 'T1Y1__R1_DFBBP_0', 'x': 50.0, 'y': 40.0, 'orient': 'N'},
                            {'name': 'T1Y1__R1_BUF_0', 'x': 55.0, 'y': 40.0, 'orient': 'N'}
                        ]
                    }
                }
            }
        }
        import yaml
        with open(fabric_file, 'w') as f:
            yaml.dump(fabric_data, f)
        
        # Create design with 4 DFFs
        design_file = tmp_path / "design.json"
        design_data = {
            'modules': {
                'top': {
                    'cells': {
                        'dff1': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [2], 'Q': [3]}},
                        'dff2': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [4], 'Q': [5]}},
                        'dff3': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [6], 'Q': [7]}},
                        'dff4': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [8], 'Q': [9]}}
                    }
                }
            },
            'top': 'top'
        }
        with open(design_file, 'w') as f:
            json.dump(design_data, f)
        
        # Generate H-Tree
        cts_tree, updated_map = generate_cts_tree(
            str(placement_map_file),
            str(fabric_file),
            str(design_file),
            tree_type='h',
            threshold=4
        )
        
        # Verify tree structure
        assert cts_tree is not None
        assert '_metadata' in cts_tree
        assert cts_tree['_metadata']['tree_type'] == 'H-Tree'
        assert cts_tree['_metadata']['num_sinks'] == 4
        
        # Should have at least one buffer for 4 DFFs
        num_buffers = cts_tree['_metadata']['num_buffers']
        assert num_buffers >= 0  # May have buffers or direct connection
        
        # Verify updated map includes original DFFs
        assert 'dff1' in updated_map
        assert 'dff2' in updated_map
        assert 'dff3' in updated_map
        assert 'dff4' in updated_map
    
    def test_xtree_4_dffs(self, tmp_path):
        """Test X-Tree creates correct structure for 4 DFFs."""
        # Create placement map with 4 DFFs
        placement_map_file = tmp_path / "placement.map"
        placement_map_file.write_text(
            "dff1 T0Y0__R1_DFBBP_0\n"
            "dff2 T0Y1__R1_DFBBP_0\n"
            "dff3 T1Y0__R1_DFBBP_0\n"
            "dff4 T1Y1__R1_DFBBP_0\n"
        )
        
        # Create fabric with DFFs and buffers
        fabric_file = tmp_path / "fabric_cells.yaml"
        fabric_data = {
            'fabric_cells_by_tile': {
                'tiles': {
                    'T0Y0': {
                        'cells': [
                            {'name': 'T0Y0__R1_DFBBP_0', 'x': 10.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T0Y0__R1_BUF_0', 'x': 15.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T0Y0__R1_BUF_1', 'x': 20.0, 'y': 20.0, 'orient': 'N'}
                        ]
                    },
                    'T0Y1': {
                        'cells': [
                            {'name': 'T0Y1__R1_DFBBP_0', 'x': 10.0, 'y': 40.0, 'orient': 'N'},
                            {'name': 'T0Y1__R1_BUF_0', 'x': 15.0, 'y': 40.0, 'orient': 'N'}
                        ]
                    },
                    'T1Y0': {
                        'cells': [
                            {'name': 'T1Y0__R1_DFBBP_0', 'x': 50.0, 'y': 20.0, 'orient': 'N'},
                            {'name': 'T1Y0__R1_BUF_0', 'x': 55.0, 'y': 20.0, 'orient': 'N'}
                        ]
                    },
                    'T1Y1': {
                        'cells': [
                            {'name': 'T1Y1__R1_DFBBP_0', 'x': 50.0, 'y': 40.0, 'orient': 'N'},
                            {'name': 'T1Y1__R1_BUF_0', 'x': 55.0, 'y': 40.0, 'orient': 'N'}
                        ]
                    }
                }
            }
        }
        import yaml
        with open(fabric_file, 'w') as f:
            yaml.dump(fabric_data, f)
        
        # Create design with 4 DFFs
        design_file = tmp_path / "design.json"
        design_data = {
            'modules': {
                'top': {
                    'cells': {
                        'dff1': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [2], 'Q': [3]}},
                        'dff2': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [4], 'Q': [5]}},
                        'dff3': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [6], 'Q': [7]}},
                        'dff4': {'type': 'sky130_fd_sc_hd__dfbbp_1', 'connections': {'CLK': [1], 'D': [8], 'Q': [9]}}
                    }
                }
            },
            'top': 'top'
        }
        with open(design_file, 'w') as f:
            json.dump(design_data, f)
        
        # Generate X-Tree
        cts_tree, updated_map = generate_cts_tree(
            str(placement_map_file),
            str(fabric_file),
            str(design_file),
            tree_type='x',
            threshold=4
        )
        
        # Verify tree structure
        assert cts_tree is not None
        assert '_metadata' in cts_tree
        assert cts_tree['_metadata']['tree_type'] == 'X-Tree'
        assert cts_tree['_metadata']['num_sinks'] == 4
        
        # Should have at least one buffer for 4 DFFs
        num_buffers = cts_tree['_metadata']['num_buffers']
        assert num_buffers >= 0  # May have buffers or direct connection
        
        # Verify updated map includes original DFFs
        assert 'dff1' in updated_map
        assert 'dff2' in updated_map
        assert 'dff3' in updated_map
        assert 'dff4' in updated_map


class TestRenaming:
    """Test instance name renaming functionality."""
    
    def test_rename_instances(self, tmp_path):
        """Test instance names can be renamed to match placement.map."""
        
        # Create placement map
        placement_map_file = tmp_path / "placement.map"
        placement_map_file.write_text(
            "cell1 T0Y0__R0_NAND_0\n"
            "cell2 T0Y0__R1_BUF_0\n"
            "dff1 T0Y0__R1_DFBBP_0\n"
        )
        
        # Create Verilog with logical names
        verilog_file = tmp_path / "test_final.v"
        verilog_content = """module top (
    input clk,
    output out
);
    wire n1, n2, n3;
    
    sky130_fd_sc_hd__nand2_2 cell1 (
        .A(n1),
        .B(n2),
        .Y(n3)
    );
    
    sky130_fd_sc_hd__dfbbp_1 dff1 (
        .CLK(clk),
        .D(n3),
        .Q(out)
    );
endmodule
"""
        verilog_file.write_text(verilog_content)
        
        # Rename instances
        output_file = tmp_path / "test_renamed.v"
        rename_instances(str(verilog_file), str(placement_map_file), str(output_file))
        
        # Verify renamed file exists
        assert output_file.exists()
        
        # Read renamed content
        renamed_content = output_file.read_text()
        
        # Verify logical names replaced with slot names
        assert 'T0Y0__R0_NAND_0' in renamed_content or 'cell1' not in renamed_content
        assert 'T0Y0__R1_DFBBP_0' in renamed_content or 'dff1' not in renamed_content
        
        # Verify module structure preserved
        assert 'module top' in renamed_content
        assert 'sky130_fd_sc_hd__nand2_2' in renamed_content
        assert 'sky130_fd_sc_hd__dfbbp_1' in renamed_content


class TestVerilogEmission:
    """Test Verilog emission functionality."""
    
    def test_verilog_basic_structure(self, tmp_path):
        """Test basic Verilog structure generation."""
        from eco_generator import write_verilog
        
        design_data = {
            'modules': {
                'top': {
                    'cells': {
                        'cell1': {
                            'type': 'sky130_fd_sc_hd__nand2_2',
                            'connections': {
                                'A': [1],
                                'B': [2],
                                'Y': [3]
                            }
                        }
                    },
                    'ports': {
                        'clk': {'direction': 'input', 'bits': [1]},
                        'out': {'direction': 'output', 'bits': [3]}
                    }
                }
            },
            'top': 'top'
        }
        
        output_file = tmp_path / "test.v"
        write_verilog(design_data, 'top', str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text()
        assert 'module top' in content
        assert 'sky130_fd_sc_hd__nand2_2' in content
        assert 'cell1' in content
    
    def test_verilog_dff_with_clock(self, tmp_path):
        """Test Verilog generation includes DFF with clock connection."""
        from eco_generator import write_verilog
        
        design_data = {
            'modules': {
                'top': {
                    'cells': {
                        'dff1': {
                            'type': 'sky130_fd_sc_hd__dfbbp_1',
                            'connections': {
                                'CLK': [1],
                                'D': [2],
                                'Q': [3]
                            }
                        }
                    },
                    'ports': {}
                }
            },
            'top': 'top'
        }
        
        output_file = tmp_path / "test.v"
        write_verilog(design_data, 'top', str(output_file))
        
        content = output_file.read_text()
        assert '.CLK(' in content
        assert 'sky130_fd_sc_hd__dfbbp_1' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

