---
title: Our Amazing Team
title_short: Team
layout: section
css_id: team

---



{% for member in site.people %}
<div class="col-sm-4">
    <div class="team-member">
        <img src="img/team/{{ member.pic }}.jpg" class="img-responsive img-circle" alt="">
        <h4>{{ member.name }}</h4>
        <p class="text-muted">{{ member.position }}</p>
        <ul class="list-inline social-buttons">
            {% for network in member.social %}
            <li>
                <a href="{{ network.url }}">
                    <i class="{{ network.title }}"></i> 
                </a>
            </li>
            {% endfor %}

        </ul>
    </div>
</div>
{% endfor %}

