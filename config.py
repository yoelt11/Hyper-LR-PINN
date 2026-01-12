import argparse
import os
import sys
import numpy as np

# Lazy parser creation - only create when get_config() is called
# This prevents interference with other argparse parsers
_parser = None

def _create_parser():
    """Create and configure the argument parser."""
    global _parser
    if _parser is None:
        _parser = argparse.ArgumentParser()
        # CPU / GPU setting
        _parser.add_argument('--device', type=str, default='cuda:0')
        _parser.add_argument('--use_cuda', type=str, default='True')
        
        _parser.add_argument('--seed', type=int, default=0)
        _parser.add_argument('--start_coeff_1', type=int, default=0, help='start point of beta range')
        _parser.add_argument('--start_coeff_2', type=int, default=0, help='start point of nu range')
        _parser.add_argument('--start_coeff_3', type=int, default=0, help='start point of rho range')
        
        _parser.add_argument('--end_coeff_1', type=int, default=0, help='end point of beta range')
        _parser.add_argument('--end_coeff_2', type=int, default=0, help='end point of nu range')
        _parser.add_argument('--end_coeff_3', type=int, default=0, help='end point of rho range')
        
        _parser.add_argument('--init_cond', type=str, default='sin_1')
        _parser.add_argument('--pde_type', type=str, default='convection')
        
        _parser.add_argument('--target_coeff_1', type=int, default=0, help='target coefficient beta')
        _parser.add_argument('--target_coeff_2', type=int, default=0, help='target coefficient nu')
        _parser.add_argument('--target_coeff_3', type=int, default=0, help='target coefficient rho')
        
        _parser.add_argument('--epoch', type=int, default=10000)
    return _parser

def get_config():
    """Get parsed command line arguments."""
    parser = _create_parser()
    return parser.parse_args()
