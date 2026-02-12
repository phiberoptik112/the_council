# Generated manually for debug_info field addition

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('council_app', '0006_add_response_validation_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='churniteration',
            name='debug_info',
            field=models.JSONField(blank=True, default=dict, help_text='Debug information including model responses and validation results'),
        ),
    ]
