# Generated by Django 4.2.16 on 2024-09-22 14:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0004_cctv_running_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="cctv",
            name="start_detection",
            field=models.BooleanField(default=False),
        ),
    ]