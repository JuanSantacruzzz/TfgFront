
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired


class UploadForm(FlaskForm):
    photo = FileField('Seleccione la imagen a estudiar:',validators=[FileRequired()])
    submit = SubmitField('Submit')

