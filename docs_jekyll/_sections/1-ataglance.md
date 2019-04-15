---
title: SoccerNet at a Glance
title_short: At a Glance
subtitle: 
layout: section
css_id: glance

---



<div class="row text-center"> 
<div class="col-md-3">
    <span class="fa-stack fa-4x">
        <i class="fa fa-circle fa-stack-2x text-primary"></i>
        <i class="fas fa-file-archive fa-stack-1x fa-inverse"></i>
    </span>
    <h4 class="service-heading">500 Games</h4>
    <p class="text-muted"></p>
</div>

<div class="col-md-3">
    <span class="fa-stack fa-4x">
        <i class="fa fa-circle fa-stack-2x text-primary"></i>
        <i class="fas fa-copy fa-stack-1x fa-inverse"></i>
    </span>
    <h4 class="service-heading"> > 5M Feature Frames</h4>
    <p class="text-muted"></p>
</div>

<div class="col-md-3">
    <span class="fa-stack fa-4x">
        <i class="fa fa-circle fa-stack-2x text-primary"></i>
        <i class="fas fa-file-alt fa-stack-1x fa-inverse"></i>
    </span>
    <h4 class="service-heading">6,637 Temporal Annotations</h4>
    <p class="text-muted"></p>
</div>

<div class="col-md-3">
    <span class="fa-stack fa-4x">
        <i class="fa fa-circle fa-stack-2x text-primary"></i>
        <i class="fas fa-chart-bar fa-stack-1x fa-inverse"></i>
    </span>
    <h4 class="service-heading">3 Event Classes</h4>
    <p class="text-muted"></p>
</div>
</div>


<p> </p>
<p> </p>
<p> </p>



<div class="row text-center"> 


<div class="col-md-4">
    <h4 class="service-heading">Goal Event</h4>
    <img src="img/dataset/TeaserGoal.png" alt="Mountain View" width="330" />
    <p class="text-muted"></p>
    <img src="img/dataset/Goal/Goal0.gif" alt="Goal" width="100" />
    <img src="img/dataset/Goal/Goal1.gif" alt="Goal" width="100" />
    <img src="img/dataset/Goal/Goal3.gif" alt="Goal" width="100" />
    <p class="text-muted"></p>
    <img src="img/dataset/Goal/Goal4.gif" alt="Goal" width="100" />
    <img src="img/dataset/Goal/Goal6.gif" alt="Goal" width="100" />
    <img src="img/dataset/Goal/Goal7.gif" alt="Goal" width="100" />
    <p class="text-muted"></p>
    <img src="img/dataset/Goal/Goal8.gif" alt="Goal" width="100" />
    <img src="img/dataset/Goal/Goal9.gif" alt="Goal" width="100" />
    <img src="img/dataset/Goal/Goal16.gif" alt="Goal" width="100" />
    <p class="text-muted"></p>
</div>


<div class="col-md-4">
    <h4 class="service-heading">Substitution Event</h4>
    <img src="img/dataset/TeaserSub.png" alt="Mountain View" width="330" />
    <p class="text-muted"></p>
    <img src="img/dataset/Substitution/Substitution0.gif" alt="Sub" width="100" />
    <img src="img/dataset/Substitution/Substitution2.gif" alt="Sub" width="100" />
    <img src="img/dataset/Substitution/Substitution5.gif" alt="Sub" width="100" />
    <p class="text-muted"></p>
    <img src="img/dataset/Substitution/Substitution6.gif" alt="Sub" width="100" />
    <img src="img/dataset/Substitution/Substitution9.gif" alt="Sub" width="100" />
    <img src="img/dataset/Substitution/Substitution11.gif" alt="Sub" width="100" />
    <p class="text-muted"></p>
    <img src="img/dataset/Substitution/Substitution16.gif" alt="Sub" width="100" />
    <img src="img/dataset/Substitution/Substitution18.gif" alt="Sub" width="100" />
    <img src="img/dataset/Substitution/Substitution19.gif" alt="Sub" width="100" />
    <p class="text-muted"></p>
</div>


<div class="col-md-4">
    <h4 class="service-heading">Card Event</h4>
    <img src="img/dataset/TeaserCard.png" alt="Mountain View" width="330" />
    <p class="text-muted"></p>
    <img src="img/dataset/Card/Card0.gif" alt="Card" width="100" />
    <img src="img/dataset/Card/Card1.gif" alt="Card" width="100" />
    <img src="img/dataset/Card/Card6.gif" alt="Card" width="100" />
    <p class="text-muted"></p>
    <img src="img/dataset/Card/Card9.gif" alt="Card" width="100" />
    <img src="img/dataset/Card/Card10.gif" alt="Card" width="100" />
    <img src="img/dataset/Card/Card17.gif" alt="Card" width="100" />
    <p class="text-muted"></p>
    <img src="img/dataset/Card/Card19.gif" alt="Card" width="100" />
    <img src="img/dataset/Card/Card20.gif" alt="Card" width="100" />
    <img src="img/dataset/Card/Card25.gif" alt="Card" width="100" />
    <p class="text-muted"></p>
</div>

<!--
<div class="col-md-4">
    <h4 class="event-heading">Goal Event</h4>
    <img src="img/dataset/TeaserGoal.png" alt="Mountain View" width="330" />

    <table>
        {% for j in (0..2) %}
        <tr>
        {% for i in (0..1) %}
            <th><img src="img/dataset/Goal/Goal{{ j | times: 5 | plus: i }}.gif" alt="Mountain View" width="150" /></th>
        {% endfor %}
        </tr>
        {% endfor %}
    </table>
</center>
</div>
 -->

<!-- 

<div class="col-md-12">
    <h2 class="event-heading">Substitution Event</h2>
    <img src="img/dataset/TeaserSub.png" alt="Mountain View" width="1000" />
</div>

<center>
    <table>
        {% for j in (0..2) %}
        <tr>
        {% for i in (0..4) %}
            <th><img src="img/dataset/Substitution/Substitution{{ j | times: 5 | plus: i }}.gif" alt="Mountain View" width="200" /></th>
        {% endfor %}
        </tr>
        {% endfor %}
    </table>
</center>


<div class="col-md-12">
    <h2 class="event-heading">Card Event</h2>
    <img src="img/dataset/TeaserCard.png" alt="Mountain View" width="1000" />
</div>

<center>
    <table>
        {% for j in (0..2) %}
        <tr>
        {% for i in (0..4) %}
            <th><img src="img/dataset/Card/Card{{ j | times: 5 | plus: i }}.gif" alt="Mountain View" width="200" /></th>
        {% endfor %}
        </tr>
        {% endfor %}
    </table>
</center>

</div> 
 -->

<!-- 
<div class="row text-center"> 
<div class="col-md-3">


{% include youtube.html id="IfC-nXlgaN" height="80%" %}

</div>
</div> -->








