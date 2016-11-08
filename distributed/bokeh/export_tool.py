from bokeh.core.properties import Int, String
from bokeh.driving import cosine
from bokeh.embed import file_html
from bokeh.io import curdoc
from bokeh.models import Tool
from bokeh.plotting import figure
from bokeh.resources import CDN


JS_CODE = """
p = require "core/properties"
ActionTool = require "models/tools/actions/action_tool"


class ExportToolView extends ActionTool.View

  initialize: (options) ->
    super(options)
    @listenTo(@model, 'change:content', @export)

  do: () ->
    # This is just to trigger an event a python callback can respond to
    @model.event = @model.event + 1

  export: () ->
    x = window.open()
    x.document.open()
    x.document.write(@model.content)
    x.document.close()

class ExportTool extends ActionTool.Model
  default_view: ExportToolView
  type: "ExportTool"
  tool_name: "Export"
  icon: "bk-tool-icon-save"

  @define {
    event:   [ p.Int,   0 ]
    content: [ p.String   ]
  }

module.exports =
  Model: ExportTool
  View: ExportToolView
"""


class ExportTool(Tool):
    __implementation__ = JS_CODE
    event   = Int(default=0)
    content = String()

    def register_plot(self, plot):
        def export_callback(attr, old, new):
            # really, export the doc as JSON
            html = 'Hello, world!' # file_html(plot, CDN, "Task Stream")
            self.content = html

        self.on_change('event', export_callback)
