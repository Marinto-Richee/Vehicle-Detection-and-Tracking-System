# Generated by Django 4.2.16 on 2024-09-28 15:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0014_remove_tracker_status_category_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="detection",
            name="unique_id",
            field=models.CharField(default="Unknown", max_length=100),
        ),
    ]
