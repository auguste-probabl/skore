{{ objname | escape | underline(line="=") }}

.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessormethod:: {{ (module.split('.')[1:] + [objname]) | join('.') }}
