---
title: Paper
title_short: Paper
subtitle: Please cite our work if you use our dataset
layout: section
css_id: publications

---


   
{% for publication in site.publications %}
<div class="col-md-4 col-sm-6 portfolio-item">
    <a href="#publication{{ publication.modal-id }}" class="portfolio-link" data-toggle="modal">
        <div class="portfolio-hover">
            <div class="portfolio-hover-content">
                <i class="fa fa-plus fa-3x"></i>
            </div>
        </div>
        <img src="img/publications/{{ publication.thumbnail }}" class="img-responsive img-centered" alt="">
    </a>
    <div class="portfolio-caption">
        <h4>{{ publication.title }}</h4>
        <p class="text-muted">
            {{ publication.subtitle }} 
        <p class="text-muted">
        </p>
            <a href="{{ publication.paper_link }}" target="_blank" class="fa fa-file-pdf"> PDF </a>
            <a href="https://youtu.be/{{ publication.yt_link }}" target="_blank" class="fab fa-youtube"> Video </a>
        </p>
        
    </div>
</div>
{% endfor %}

