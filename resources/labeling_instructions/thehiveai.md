# Guidelines
## Overview
We seek to annotate gray scale frames via labeled bounding boxes of **7 categories**: 
`trucks, bicycles, persons, train/tram, bus, cars, motorcycles`. 
### Tips
* The frames have in general low resolution (240x180). Nevertheless they are all taken in an **urban environment** which usually gives context to objects that might be under bad illumination.
* **Train/tram** should include *trains* and *trams*.
* The frames come from Zurich where **trams** look like [this](https://www.google.ch/search?biw=1855&bih=965&tbm=isch&sa=1&ei=61cJWtOzEYOuau6sqogG&q=zurich+trams&oq=zurich+trams&gs_l=psy-ab.3..0l2j0i24k1l2.8724.8724.0.9310.1.1.0.0.0.0.64.64.1.1.0....0...1.1.64.psy-ab..0.1.64....0.Vwq02vZv6xc).
* **Truck or car?** According to common knowledge trucks look like [this](https://www.google.ch/search?biw=1855&bih=1085&tbm=isch&sa=1&ei=eg4MWovzNc6x0gWQ0YaIAg&q=trucks+images&oq=trucks+images&gs_l=psy-ab.3..0l4j0i5i30k1l6.3127.4339.0.4580.7.7.0.0.0.0.97.529.6.6.0....0...1.1.64.psy-ab..1.6.527...0i67k1.0.zrOkpNA9JA8), which also includes ambulances and similar vehicles.
* If the number of bounding boxes is more that the maximum, discard the smaller ones. In general **we care much more about the bigger objects**.
* If multiple objects of the same type are **overlapping** (parked bikes for examples), a single bounding box is sufficient. If they are only **partially overlapping** please distinguish among them. 
---------------
* Examples of annotated frames with partially **overlapping objects**: 

<img src="https://s3.amazonaws.com/docs.thehive.ai/guideline_images/d5386e40-d814-4835-91fd-f8d69fc61238" height="150"/>
<img src="https://s3.amazonaws.com/docs.thehive.ai/guideline_images/e988ce71-02be-48dc-9918-534815645b64" height="150"/>

---------------
* **People riding** bikes or motorcycles should be annotate like this:


<img src="https://s3.amazonaws.com/docs.thehive.ai/guideline_images/9c0a0122-6e77-4b45-bba1-758055fcca76" height="150"/>
<img src="https://s3.amazonaws.com/docs.thehive.ai/guideline_images/b969c864-a5d7-4151-871a-37e70c351d58" height="150"/>

---------------
* Also **partial objects** should be annotated if one is able to clearly tell the category of the object, examples:


<img src="https://s3.amazonaws.com/docs.thehive.ai/guideline_images/4649288f-8cb9-487e-a4ab-692457d362bb" height="150"/>