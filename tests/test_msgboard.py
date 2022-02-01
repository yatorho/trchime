from trchime.call.view import MessageBoard


mb = MessageBoard()
mb.add_horizontal_line()
mb.add_text(1, "parameter:", width = 15)
mb.add_text(1, str(12800), width = 15)
mb.add_horizontal_line(full = 2)

mb.show()




