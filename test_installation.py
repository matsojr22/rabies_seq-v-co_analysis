#!/usr/bin/env python3
"""
Test script to verify that all required packages are installed correctly.
Run this script after installation to ensure everything is working.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def main():
    """Test all required packages."""
    print("Testing rabies analysis environment...")
    print("=" * 50)
    
    # Core packages
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("sklearn", "scikit-learn"),
        ("openpyxl", "openpyxl"),
        ("seaborn", "seaborn"),
    ]
    
    # Optional packages
    optional_packages = [
        ("plotly", "plotly"),
    ]
    
    print("\nCore packages:")
    all_core_passed = True
    for module, package in packages:
        if not test_import(module, package):
            all_core_passed = False
    
    print("\nOptional packages:")
    for module, package in optional_packages:
        test_import(module, package)
    
    print("\n" + "=" * 50)
    
    if all_core_passed:
        print("✓ All core packages are installed successfully!")
        print("\nYou can now run the analysis:")
        print("python rabies_compare_2.py")
    else:
        print("✗ Some core packages are missing.")
        print("Please install them using:")
        print("pip install -r requirements.txt")
        print("or")
        print("conda env create -f environment.yml")
        sys.exit(1)

if __name__ == "__main__":
    main()

