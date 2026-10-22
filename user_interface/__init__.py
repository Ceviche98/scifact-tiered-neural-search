import os
import streamlit.components.v1 as components

# Create the _component_func
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend")

_neumorphic_switch = components.declare_component("neumorphic_switch", path=build_dir)

def neumorphic_switch(key=None, default=False):
    """
    Returns True if Pro Tier (Dark Mode / Crown), False if Fast Tier (Light Mode / Lightning)
    """
    component_value = _neumorphic_switch(key=key, default=default)
    if component_value is None:
        return default
    return component_value
