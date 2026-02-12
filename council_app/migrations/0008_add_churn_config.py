# Generated for ChurnConfig model

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('council_app', '0007_add_debug_info_to_churn_iteration'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChurnConfig',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('num_predict', models.PositiveIntegerField(blank=True, help_text='Max tokens to generate (blank=unlimited)', null=True)),
                ('keep_alive', models.CharField(blank=True, help_text='e.g. 5m, 0, blank=default', max_length=20)),
                ('num_ctx', models.PositiveIntegerField(blank=True, help_text='Context window (blank=model default)', null=True)),
                ('sequential_models', models.BooleanField(default=True, help_text='Run models one at a time (better for Pi)')),
                ('max_content_chars', models.PositiveIntegerField(blank=True, help_text='Max chars of content in prompts (blank=no limit)', null=True)),
                ('max_synthesis_response_chars', models.PositiveIntegerField(blank=True, help_text='Max chars per response in synthesis', null=True)),
                ('use_streaming', models.BooleanField(default=False, help_text='Stream Ollama response to avoid timeout')),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Churn Configuration',
                'verbose_name_plural': 'Churn Configuration',
            },
        ),
    ]
