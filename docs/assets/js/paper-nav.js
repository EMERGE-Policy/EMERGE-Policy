(function () {
	'use strict';

	var nav = document.querySelector('.paper-nav');

	if (!nav) {
		return;
	}

	var links = Array.prototype.slice.call(
		nav.querySelectorAll('a[href^="#"]')
	);
	var sections = links
		.map(function (link) {
			return document.querySelector(link.getAttribute('href'));
		})
		.filter(Boolean);

	function setActive(id) {
		links.forEach(function (link) {
			var isActive = link.getAttribute('href') === '#' + id;
			link.classList.toggle('is-active', isActive);

			if (isActive) {
				link.setAttribute('aria-current', 'location');
			} else {
				link.removeAttribute('aria-current');
			}
		});
	}

	links.forEach(function (link) {
		link.addEventListener('click', function () {
			setActive(link.getAttribute('href').slice(1));
		});
	});

	if ('IntersectionObserver' in window) {
		var observer = new IntersectionObserver(function (entries) {
			var visibleSections = entries
				.filter(function (entry) {
					return entry.isIntersecting;
				})
				.sort(function (a, b) {
					return a.boundingClientRect.top - b.boundingClientRect.top;
				});

			if (visibleSections.length) {
				setActive(visibleSections[0].target.id);
			}
		}, {
			rootMargin: '-18% 0px -62% 0px',
			threshold: [0, 0.1, 0.5]
		});

		sections.forEach(function (section) {
			observer.observe(section);
		});
	}

	if (window.location.hash) {
		setActive(window.location.hash.slice(1));
	}
})();
