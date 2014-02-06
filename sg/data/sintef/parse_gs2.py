#!/usr/bin/python

import os

from sg.utils.timer import SimpleTimer

_KEYVAL_SEPARATOR = "="
_VALUE_OPEN = "<"
_VALUE_CLOSE = ">"
_ENTRY_INDICATOR = "#"
_HEADING_INDICATOR = "##"

def _is_element_type(line, element_indicator):
    return len(line) >= len(element_indicator) and \
        line[:len(element_indicator)] == element_indicator

def _is_entry(line):
    return _is_element_type(line, _ENTRY_INDICATOR)

def _is_heading(line):
    return _is_element_type(line, _HEADING_INDICATOR)

def _make_section(heading_line):
    return (heading_line[2:-1], dict())

def _close_value(stream):
    line = ""
    for next_line in stream:
        line += next_line
        if _VALUE_CLOSE in next_line:
            return line
    
def _find_value(line, stream):
    if _VALUE_OPEN in line:
        if not _VALUE_CLOSE in line:
            line += _close_value(stream)
        contents = line[line.index(_VALUE_OPEN) + 1:line.index(_VALUE_CLOSE)]
        return contents.split()
    return [line[:-1]]

def _split_line(line, stream):
    try:
        separator_idx = line.index(_KEYVAL_SEPARATOR)
        key = line[1:separator_idx]
        value = _find_value(line[separator_idx + 1:], stream)
        return (key, value)
    except ValueError:
        raise ValueError("Error while splitting a line into key and value "
                         "constituents! Key/value separator sign (" +
                         _KEYVAL_SEPARATOR + ") not found in input line:\n\t" +
                         line)
    
def _add_key_value(section, line, stream):
    key, value = _split_line(line, stream)
    contents = section[1]
    if key in contents:
        raise ValueError("Duplicate key '" + key +
                         "' in section '" + section[0])
    contents[key] = value

def _find_first_heading(stream):
    for line in stream:
        if _is_heading(line):
            return line

def section_generator(stream):
    """Iterator generator. Each call to the generated iterator will return the
    next section of a GS2 file. A section is represented as a two-element
    tuple, where the first element is the section heading and the second
    element is a dictionary. The dictionary holds the entries as key/value
    pairs. Each value is a list."""
    line = _find_first_heading(stream)
    if line is None:
        return
    section = _make_section(line)
    for line in stream:
        if _is_heading(line):
            yield section
            section = _make_section(line)
        elif _is_entry(line):
            _add_key_value(section, line, stream)
    yield section

def parse_file(path):
    """Parse a GS2 file and return a list where each element is a section in
    the GS2 file. See section_generator for info on the format of each
    section."""
    with open(path, "r") as f:
        return [section for section in section_generator(f)]

def parse_all_generator(pathfile):
    """Given a file containing a list of GS2 files, parse all the GS2 files one
    after the other."""
    with open(pathfile) as paths:
        for path in paths:
            path = path[:-1]
            yield (path, parse_file(path))


if __name__ == '__main__':
    from unittest import main
    main(module='test_'+__file__[:-3])
