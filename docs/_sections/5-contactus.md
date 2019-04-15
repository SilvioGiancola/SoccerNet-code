---
title: Contact us
title_short: Contact
subtitle: Please fill in the from below and briefly describe your issue/use-case. <br> You may also use this form to contact us with any further questions.
layout: section
css_id: contact

---




<div class="col-lg-12">
    <form action="https://www.enformed.io/{{ site.enformed_token }}" method="POST" > 

        <div class="col-md-6">

            <div class="control-group form-group">
                <input name="name" type="text" class="form-control" placeholder="Your Name *" required data-validation-required-message="Please enter your name."/>
                <p class="help-block text-danger"></p>
            </div>

            <div class="control-group form-group">
                <input name="email" type="text" class="form-control" placeholder="Your Email *" id="email" required data-validation-required-message="Please enter your email address.">
                <p class="help-block text-danger"></p>
            </div>


        </div>

        <div class="col-md-6">

            <div class="control-group form-group">
                <textarea name="message" type="text" class="form-control" placeholder="Your Message *" id="message" required data-validation-required-message="Please enter a message."></textarea>
                <p class="help-block text-danger"></p>
            </div>

        </div>

    <!--     <div class="col-lg-12 text-center">

            <h3 class="section-subheading text-muted">
                <input name="EULA" type="checkbox" class="btn btn-xl" placeholder="Your Email *" id="email" required data-validation-required-message="Please accept."> 
                        I agree the <a href="{{ site.url }}{{ site.baseurl }}EULA.pdf" target="_blank">EULA</a> 
            </h3>
        </div> -->

        <div class="col-lg-12 text-center">
            <button type="submit" class="btn btn-xl" onclick="alert('Thank you for contacting us, we will answer as soon as possible.')">Send Message</button>
        </div>

        <input type="hidden" name="*honeypot" />
        <input type="hidden" name="*subject"  value="SoccerNet Download Request" />            
        <input type="hidden" name="*formname" value="SoccerNet" />
        <input type="hidden" name="*reply"    value="email" />
        <input type="hidden" name="*redirect" value="{{ site.url }}{{ site.baseurl }}" />


    </form>
</div> 


<!-- 

<div class="col-lg-12">
    <form action="/SoccerNet/download.html" method="get"> 

        <div class="col-md-6">

            <div class="control-group form-group">
                <input name="name" type="text" class="form-control" placeholder="Your Name *" required data-validation-required-message="Please enter your name."/>
                <p class="help-block text-danger"></p>
            </div>

            <div class="control-group form-group">
                <input name="email" type="text" class="form-control" placeholder="Your Email *" id="email" required data-validation-required-message="Please enter your email address.">
                <p class="help-block text-danger"></p>
            </div>


        </div>

        <div class="col-md-6">

            <div class="control-group form-group">
                <textarea name="message" type="text" class="form-control" placeholder="Your Message *" id="message" required data-validation-required-message="Please enter a message."></textarea>
                <p class="help-block text-danger"></p>
            </div>

        </div>

        <div class="col-lg-12 text-center">

            <h3 class="section-subheading text-muted">
                <input name="EULA" type="checkbox" class="btn btn-xl" required data-validation-required-message="Please accept."> 
                        I agree the <a href="{{ site.url }}{{ site.baseurl }}EULA.pdf" target="_blank">EULA</a> 
            </h3>
        </div>

        <div class="col-lg-12 text-center">
            <button type="submit" class="btn btn-xl" onclick="alert('Hello World!')">Send Request</button>
        </div>


        <input type="hidden" name="*redirect" value="{{ site.url }}{{ site.baseurl }}" />


    </form>
</div>  -->


<!-- <div class="col-lg-12">
    <form action="https://goo.gl/forms/0w85nU7qG4j0613f2" target="_blank">
        <div class="col-lg-12 text-center">
            <button type="submit" class="btn btn-xl">Request Access</button>
        </div>
    </form>
</div> -->


<!-- <div class="col-lg-12">
<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform?embedded=true" width="760" height="500" frameborder="0" marginheight="0" marginwidth="0">Loading...</iframe>
</div>  -->
<!-- </section> -->




