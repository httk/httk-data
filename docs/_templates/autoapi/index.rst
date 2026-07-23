API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :titlesonly:

{% for page in pages %}
   {% set state = namespace(has_rendered_parent=false) %}
   {% for candidate in pages %}
      {% if candidate.id != page.id and page.id.startswith(candidate.id ~ ".") %}
         {% set state.has_rendered_parent = true %}
      {% endif %}
   {% endfor %}
   {% if not state.has_rendered_parent %}
   {{ page.include_path }}
   {% endif %}
{% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
