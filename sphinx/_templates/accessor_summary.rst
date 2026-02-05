{% for class_info in classes %}
.. accessor_dropdown:: {{ class_info.name }}
   :doc: "{{ class_info.doc | e }}"
   :full_name: {{ class_info.full_name }}
   :uri: {{ class_info.uri }}

   {% for accessor_name, accessor_data in class_info.accessors.items() %}
   **{{ accessor_name.capitalize() }}**: :class:`{{ class_info.name }}.{{ accessor_name }}`

   .. list-table::
      :widths: 30 70
      {% for method_name, method_doc in accessor_data.methods %}
      * - :func:`~{{ class_info.name }}.{{ accessor_name }}.{{ method_name }}`
        - {{ method_doc }}
      {% endfor %}
   {% endfor %}
{% endfor %}
